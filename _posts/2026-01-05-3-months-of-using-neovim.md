---
layout: post
title: 3 months of using Neovim
date: 2026-01-05
comments: false
tags: [IDE, neovim, programming]
archive: false
---

I have been using [neovim](https://neovim.io/) for the past 3 months not because I wanted to switch to a new IDE but because my org has a very bureaucratic process to open [VS Code](https://code.visualstudio.com/) or [Cursor](https://cursor.sh/) over SSH, so instead what I usually do is spin up an [SSH terminal over http](/2025/10/ssh-over-tor/) and access it from my personal machine and use neovim as my editor. I will first try to introduce you to neovim and the real juice of it the plugin manager [lazy.nvim](https://github.com/folke/lazy.nvim) and then list down my frustrations with it. If you want to learn neovim, I would recommend to start with neovim [:Tutor](https://neovim.io/doc/user/nvim.html#nvim-intro)

## Why Neovim

Neovim just works. It's fast, it's everywhere, and it doesn't need a GUI. That's it. But after 3 months of using it I have come to appreciate some of its features that I honestly didn't expect

Neovim is a fork of [Vim](https://en.wikipedia.org/wiki/Vim_(text_editor)) that focuses on extensibility and usability. The key difference is that neovim uses [Lua](https://www.lua.org/) for configuration instead of vimscript which is honestly a huge improvement. Lua is actually a programming language you can understand without wanting to throw your laptop. A lot of good games like [Noita](https://en.wikipedia.org/wiki/Noita_(video_game)) are also written in lua

## Getting Started

Installing neovim is straightforward on most systems:

```bash
# Ubuntu/Debian
sudo apt install neovim

# Mac
brew install neovim

# From source (for latest version)
git clone https://github.com/neovim/neovim.git
cd neovim && make CMAKE_BUILD_TYPE=Release
sudo make install
```

The main configuration file lives at `~/.config/nvim/init.lua`. If you're coming from vim it's similar to `.vimrc` but in lua. Here's a minimal setup to get started:

```lua
-- init.lua
vim.opt.number = true           -- line numbers
vim.opt.relativenumber = true   -- relative line numbers
vim.opt.tabstop = 4             -- tab width
vim.opt.shiftwidth = 4          -- indent width
vim.opt.expandtab = true        -- spaces instead of tabs
vim.opt.smartindent = true      -- smart indentation
vim.opt.termguicolors = true    -- true color support
vim.opt.clipboard = "unnamedplus" -- system clipboard

-- leader key
vim.g.mapleader = " "
```

## Plugin Management with lazy.nvim

This is where neovim really shines. The plugin ecosystem is massive. I originally used [packer.nvim](https://github.com/wbthomason/packer.nvim) but it's now unmaintained so I switched to [lazy.nvim](https://github.com/folke/lazy.nvim) which is faster and has a nicer UI. Here's how you bootstrap it:

```lua
-- Bootstrap lazy.nvim
local lazypath = vim.fn.stdpath("data") .. "/lazy/lazy.nvim"
if not vim.loop.fs_stat(lazypath) then
  vim.fn.system({
    "git", "clone", "--filter=blob:none",
    "https://github.com/folke/lazy.nvim.git",
    "--branch=stable", lazypath,
  })
end
vim.opt.rtp:prepend(lazypath)

require("lazy").setup({
  -- your plugins go here
})
```

Adding plugins is pretty simple. For example to add the [telescope fuzzy finder](https://github.com/nvim-telescope/telescope.nvim) and [treesitter for syntax highlighting](https://github.com/nvim-treesitter/nvim-treesitter)

```lua
require("lazy").setup({
  {
    "nvim-telescope/telescope.nvim",
    dependencies = { "nvim-lua/plenary.nvim" },
    keys = {
      { "<leader>ff", "<cmd>Telescope find_files<cr>", desc = "Find Files" },
      { "<leader>fg", "<cmd>Telescope live_grep<cr>", desc = "Live Grep" },
    },
  },
  {
    "nvim-treesitter/nvim-treesitter",
    build = ":TSUpdate",
    config = function()
      require("nvim-treesitter.configs").setup({
        ensure_installed = { "lua", "python", "javascript", "bash" },
        highlight = { enable = true },
      })
    end,
  },
})
```

## My Setup

After 3 months of tweaking here's what I actually use on a daily basis:

- **[telescope.nvim](https://github.com/nvim-telescope/telescope.nvim)** - Fuzzy finder for everything. Files, grep, buffers, git commits. This is probably the most essential plugin
- **[nvim-lspconfig](https://github.com/neovim/nvim-lspconfig)** - LSP configuration for autocomplete and go-to-definition. Works well with pyright for python
- **[nvim-cmp](https://github.com/hrsh7th/nvim-cmp)** - Autocompletion engine. Pairs well with lspconfig
- **[lualine.nvim](https://github.com/nvim-lualine/lualine.nvim)** - Status line that actually looks good
- **[catppuccin](https://github.com/catppuccin/nvim)** - Color scheme because the default is shit
- **[gitsigns.nvim](https://github.com/lewis6991/gitsigns.nvim)** - Git integration in the gutter

My approximate keybindings that I use most frequently:

```lua
-- Quick navigation
vim.keymap.set("n", "<leader>e", vim.cmd.Ex)  -- file explorer
vim.keymap.set("n", "<C-d>", "<C-d>zz")        -- center after scroll
vim.keymap.set("n", "<C-u>", "<C-u>zz")        -- center after scroll

-- LSP
vim.keymap.set("n", "gd", vim.lsp.buf.definition)
vim.keymap.set("n", "K", vim.lsp.buf.hover)
vim.keymap.set("n", "<leader>rn", vim.lsp.buf.rename)
```

Here is my [neovim config](https://gist.github.com/martianlantern/be52c6b8bc7a51f6873dc91b99a73975)

## Things that suck

**1. The learning curve is brutal**

I thought I was productive in the first week. I was not. It took me a solid month before I stopped reaching for the mouse or accidentally exiting insert mode. The modal editing paradigm is fundamentally different and your muscle memory actively fights you.

**2. Configuration is a time sink**

Every hour I save by being efficient in neovim I have probably spent 10 hours configuring it. The init.lua file grows and grows. Something breaks after an update and you spend 2 hours debugging why treesitter suddenly stopped working. It's endless.

**3. LSP setup is painful**

Getting LSP to work properly requires setting up the language server, the client configuration, the keybindings, the autocompletion. Each step can fail silently. In VSCode you install an extension and it just works. In neovim you're reading github issues at 2am trying to figure out why pyright isn't detecting your virtual environment.

**4. Plugin compatibility issues**

Plugins sometimes don't play well together. Lazy loading can break things in unexpected ways. A plugin update might break another plugin. You become a dependency manager for your text editor.

**5. No remote development the way VSCode does it**

VSCode has this amazing remote SSH extension where your local VSCode connects to a remote server and it feels native. Neovim over SSH is just neovim over a slow connection. There's no local rendering magic happening.

## My Copium

What I will emphasize is that neovim has made me more aware of my editing patterns. The modal editing forces you to think about what you're doing. I've become faster at certain operations like multi-line edits and navigating large files