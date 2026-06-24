---
layout: default
title: "Blog"
permalink: /blog/
---

<div class="row g-5 mb-5">
  <div class="col-md-12">
    <h3 class="fw-bold border-bottom pb-3 mb-5">Blog</h3>
    {% assign posts = site.blog | sort: 'date' | reverse %}
    {% for post in posts %}
      <p class="mb-2"><a href="{{ site.github.url }}{{ post.url }}" style="color: #2a6496 !important; text-decoration: none; font-weight: 600;">{{ post.title }}</a> <span style="color: #555;">— {{ post.date | date: "%B %-d, %Y" }}</span></p>
    {% endfor %}
  </div>
</div>
