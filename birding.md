---
layout: default
title: Birding
---

<style>
  figcaption {
    text-align: center;
    font-size: 1rem;
  }

  .name {
    font-weight: bold;
  }
</style>

<table>
  {% for image in site.data.birding.images %}
  {% assign filename = image.name | append: ".png" %}
  {% assign caption = image.name | replace: "_", " " | capitalize %}
  {% assign scientific_name = image.scientific_name %}
  <tr>
    <td>
      <figure>
        <img src="{{ site.url }}/assets/images/birding/{{ filename }}" alt="{{ caption }}">
        <figcaption>
          <span class="name"><em>{{ caption }}</em></span>
          (<span class="scientific-name"><em>{{ scientific_name }}</em></span>)
        </figcaption>
      </figure>
    </td>
    {% cycle "", "</tr><tr>" %}
  </tr>
  {% endfor %}
</table>