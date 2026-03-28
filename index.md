---
layout: default
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<header style="display: flex; align-items: center; gap: 2rem; margin-bottom: 3rem;">
    <img src="/assets/images/personalPage/profile_picture.jpg" 
         alt="Jordi Alonso" 
         style="width: 120px; height: 120px; border-radius: 50%; object-fit: cover; border: 2px solid #eee;">
    
    <div>
        <h1 style="margin: 0; font-size: 2rem;">Jordi Alonso</h1>
        <p style="margin: 0.5rem 0 0; color: #666; font-style: italic;">
            Interesting bits in applied mathematics & machine learning.
        </p>
    </div>
</header>

<hr style="border: 0; border-top: 1px solid #eee; margin: 2rem 0;">

<section>
    {% for tag in site.tags %}
      <div class="tag-group" style="margin-bottom: 2rem;">
        <h3 style="text-transform: capitalize; color: #2a7ae2; border-bottom: 1px solid #fafafa; display: inline-block;">
            #{{ tag[0] }}
        </h3>
        <ul style="list-style: none; padding-left: 0;">
          {% for post in tag[1] %}
            <li style="margin-bottom: 0.8rem; display: flex; align-items: baseline;">
    <span style="font-family: monospace; color: #888; margin-right: 15px; font-size: 0.85rem; min-width: 100px;">
        {{ post.date | date: "%b %d, %Y" }}
    </span>
    <a href="{{ post.url }}" class="post-link">
        {{ post.title }}
    </a>
</li>

<style>
    /* Vibrant link color */
    .post-link {
        text-decoration: none;
        font-weight: 600;
        color: #0070f3; /* Bright 'Vercel' style blue */
        transition: color 0.2s ease;
        font-size: 0.95rem; /* Smaller, cleaner link size */
    }

    /* Interactive hover effect */
    .post-link:hover {
        color: #ff0080; /* Pops to a bright pink/magenta on hover */
        text-decoration: underline;
    }

    /* Tag styling */
    .tag-title {
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.95rem;
        color: #444;
        background: #f0f0f0;
        padding: 2px 8px;
        border-radius: 4px;
    }
</style>
          {% endfor %}
        </ul>
      </div>
    {% endfor %}
</section>