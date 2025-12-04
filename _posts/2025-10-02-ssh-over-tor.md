---
layout: post
title: SSH over Tor
date: 2025-10-02
comments: false
tags: [ssh, tor, networking, security]
archive: false
---

I previously discussed how we can [setup an ssh server](/2025/01/setting-up-ssh-server/) between a host and a client. We had to configure our router and it's firewall to enable port forwarding over TCP:22. Sometimes we don't even have access to configure the router or are behind a "dumb" [firewall](https://en.wikipedia.org/wiki/Firewall_(computing)) (in my company). In such cases attempting to connect via ssh will result in time out errors. Here I discuss how we can setup an ssh which is hidden behind the tor network.

**Table of Contents:**
- [A Brush up on Firewalls](#a-brush-up-on-firewalls)
- [Tor Network](#tor-network)
- [Establishing SSH connection over Tor](#establishing-ssh-connection-over-tor)
- [Hiding your Tor Service](#hiding-your-tor-service)
- [Speeding up Tor Connection](#speeding-up-tor-connection)

## A Brush up on Firewalls

Firewalls control which types of packets (TCP, UDP, etc) can go to and from which ports, ip address and applications. Luckily the firewall in my company is configured to allow internet access by outbound TCP over port 80 for HTTP but to block everything else. Tor uses TLS over TCP with it's own protocol and listens to port 443 which makes it ideal to hide SSH connections.

## Tor Network

[Tor](https://en.wikipedia.org/wiki/Tor_(network)) is an overlay network that is specifically designed for anonymous communication. The Tor network maintains a big index of all the nodes in the network and their IPs and public keys. When we want to send a packet, our client picks a random route from those node network and wraps the data in several layers of encryption that can each be only opened by the next node in the route (in the manner of [matryoshka dolls](https://en.wikipedia.org/wiki/Matryoshka_doll)). This is the fundamental principle behind [onion routing](https://en.wikipedia.org/wiki/Onion_routing). This means that any observer can see our packets go from our computer to a Tor router but can't decrypt the next packet the Tor router then forwards to. That is why Tor also acts as a [SOCKS](https://en.wikipedia.org/wiki/SOCKS) proxy. Because it is just an open network like any other we can configure it to provide access to our SSH server.

### Configuring Tor

Tor operates on TCP over 443 with it's own protocol so any firewalls or SSH blockers or network packets inspector allow Tor to work without any hiccups. Infact as long as we can do outbound network connection Tor will work just fine. To configure tor with ssh on the server we can first install and enable them both on linux:

```bash
apt update && apt install openssh-server tor torsocks -y
systemctl enable --now tor
```

Now we configure tor where to save our ssh service's state and keys and also our hostname (.onion address). Because this is unique to each service and users, deleting this will change our .onion address if we ever restart our server. To prevent I discuss a way to keep this static. We force tor use V3 which is cryptographically stronger and has much better [service descriptors](https://www.lagomframework.com/documentation/1.6.x/java/ServiceDescriptors.html). Then lastly we forward our local TCP:22 port which we use for SSH to the onion network as virtual port 22.

To do this here are the changes we append to `/etc/tor/torrc`:

```
HiddenServiceDir /var/lib/tor/ssh_service
HiddenServiceVersion 3
HiddenServicePort 22 127.0.0.1:22
```

Restart tor for changes to take effect:

```bash
systemctl restart tor
```

Then we can read our hostname on the onion network:

```bash
cat /var/lib/tor/ssh_service/hostname
# >> uytzpoijvz5okghzzzlvisenfsencsykdgtxljh4iq7scqubgmiw2mvyd.onion
```

### Configuring SSH

Configuring SSH proceeds exactly the same way as we discussed in [setting up an SSH server](/2025/01/setting-up-ssh-server/). Keep in mind to make it secure by adding Auth keys. Then finally ensure SSH is up and running again:

```bash
systemctl restart ssh
```

## Establishing SSH connection over Tor

To connect with SSH over Tor we need to also install and enable tor and ssh on the client:

```bash
apt update
apt install openssh-server tor torsocks -y
systemctl enable ssh tor
```

After this, connecting to server is very straight forward and similar to setting up any normal ssh connection, we use the onion address from earlier section. We also tell SSH to use the SOCKS5 proxy using netcat while connecting. It then performs a CONNECT at 127.0.0.1:9050 before relaying traffic to the host and port.

Add this to your `~/.ssh/config`:

```
Host tor_ssh_server
    HostName uytzpoijvz5okghzzzlvisenfsencsykdgtxljh4iq7scqubgmiw2mvyd.onion
    User root
    ProxyCommand /usr/bin/nc -x 127.0.0.1:9050 -X 5 %h %p
    IdentityFile ~/.ssh/tor_ssh_private_key
```

If all goes well you will be successfully connected to you SSH server over the Tor network. You might notice the response time is slightly slow which is due to the additional hops over the nodes networks the packets are performing to maintain anonymity.

## Hiding your Tor Service

Apart from adding authentication at the SSH level we can make our connection more secure by adding an additional layer of security at the Tor layer. We can restrict who can even see our service with [client auth](https://community.torproject.org/onion-services/advanced/client-auth/). This gates the services to only those clients that has a valid matching keys. Without the private keys Tor won't be able to fetch our service descriptors and our service will be effectively invisible to others.

## Speeding up Tor Connection

When we see the debug logs of our SSH connection we can see that:

```
debug3: obfuscate_keystroke_timing: starting: interval ~20ms
... chaff packets sent
```

To speed up the connection, you can add these options to your SSH config:

```
ObscureKeystrokeTiming no
IPQoS none
Compression no
```

For even faster connections (at the cost of some anonymity), you can configure Tor to use single-hop mode in `/etc/tor/torrc`:

```
HiddenServiceNonAnonymousMode 1
HiddenServiceSingleHopMode 1
```

Note: Single-hop mode reduces anonymity but significantly improves latency for trusted use cases.

