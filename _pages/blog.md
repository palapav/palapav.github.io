---
title: ""
permalink: /blog/
layout: archive
author_profile: true
---

<div class="blog-header">
  <h1>Blog</h1>
  <p class="blog-description">This space is where I document ideas that don’t quite fit into formal publications: side projects, engineering builds, early-stage research directions, and personal deep-dives. Some posts are informal walkthroughs, others are research-style notes capturing what I learned while prototyping new methods, testing out a hunch, or chasing an idea I couldn’t leave alone.</p>
</div>

<div class="blog-posts">
  {% for post in site.posts %}
    <article class="blog-post">
      <header class="post-header">
        <h2 class="post-title">
          <a href="{{ post.url }}">{{ post.title }}</a>
        </h2>
        <div class="post-meta">
          <time datetime="{{ post.date | date_to_xmlschema }}">
            {{ post.date | date: "%B %d, %Y" }}
          </time>
          {% if post.tags %}
            <span class="post-tags">
              {% for tag in post.tags %}
                <span class="tag">{{ tag }}</span>
              {% endfor %}
            </span>
          {% endif %}
        </div>
      </header>
      
      {% if post.excerpt %}
        <div class="post-excerpt">
          {{ post.excerpt }}
          <a href="{{ post.url }}" class="read-more">Read more →</a>
        </div>
      {% endif %}
    </article>
  {% endfor %}
</div>

<style>
.blog-header {
  text-align: center;
  margin-bottom: 3rem;
  padding: 2rem 0;
  border-bottom: 1px solid #eaeaea;
}

.blog-header h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: #333;
}

.blog-description {
  font-size: 1.2rem;
  color: #666;
  margin: 0;
}

.blog-posts {
  max-width: 800px;
  margin: 0 auto;
}

.blog-post {
  margin-bottom: 3rem;
  padding-bottom: 2rem;
  border-bottom: 1px solid #f0f0f0;
}

.blog-post:last-child {
  border-bottom: none;
}

.post-title {
  font-size: 1.8rem;
  margin-bottom: 0.5rem;
}

.post-title a {
  color: #333;
  text-decoration: none;
}

.post-title a:hover {
  color: #007acc;
}

.post-meta {
  font-size: 0.9rem;
  color: #666;
  margin-bottom: 1rem;
}

.post-tags {
  margin-left: 1rem;
}

.tag {
  background: #f0f0f0;
  padding: 0.2rem 0.6rem;
  border-radius: 3px;
  font-size: 0.8rem;
  margin-right: 0.5rem;
}

.post-excerpt {
  color: #555;
  line-height: 1.6;
}

.read-more {
  color: #007acc;
  text-decoration: none;
  font-weight: 500;
}

.read-more:hover {
  text-decoration: underline;
}
</style> 