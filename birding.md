---
layout: default
title: Birding
---

<div class="col-sm-12">
  <div class="page-title">
    {{ page.title }}
  </div>
  <div class="page-content birding-gallery">
    <div class="birding-grid">
      {% for image in site.data.birding.images %}
      {% assign filename = image.name | append: ".png" %}
      {% assign caption = image.name | replace: "_", " " | capitalize %}
      {% assign scientific_name = image.scientific_name %}
      <figure class="birding-item">
        <img src="{{ site.url }}/assets/images/birding/{{ filename }}" alt="{{ caption }}">
        <figcaption>
          <span class="bird-name"><em>{{ caption }}</em></span>
          <span class="bird-scientific">(<em>{{ scientific_name }}</em>)</span>
        </figcaption>
      </figure>
      {% endfor %}
    </div>
  </div>
</div>

<style>
.birding-gallery {
  padding-top: 20px;
}

.birding-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 2rem;
  margin-top: 1rem;
}

.birding-item {
  margin: 0;
  display: flex;
  flex-direction: column;
}

.birding-item img {
  width: 100%;
  height: auto;
  display: block;
  margin-bottom: 0.75rem;
}

.birding-item figcaption {
  text-align: center;
  font-size: 14px;
  line-height: 1.5;
  color: #353535;
}

.bird-name {
  font-weight: 600;
  display: block;
  margin-bottom: 0.2rem;
}

.bird-scientific {
  font-size: 13px;
  color: #888;
}

/* Dark mode support */
[data-theme="dark"] .birding-item figcaption {
  color: #c9d1d9;
}

[data-theme="dark"] .bird-name {
  color: #c9d1d9;
}

[data-theme="dark"] .bird-scientific {
  color: #8b949e;
}

/* Responsive */
@media (max-width: 768px) {
  .birding-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
}
</style>