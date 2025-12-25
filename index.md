---
layout: default
---


<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>


<div style="display:flex;">
    <div style="flex:1;">
        <img src="/assets/images/personalPage/profile_picture.jpg" alt="Profile Picture" width="150px">
    </div>
    <div style="flex:2;">
        <p><strong>Hello, and welcome to my online notebook!!</strong></p>
        <p>Here I gather some notes on different topics</p>
        <p>Drop me a message: JordiAlonso@proton.me </p>
        
    </div>
</div>


<div style="height:40px;"></div>  <!-- adds space -->

{% for tag in site.tags %}
  <h3>{{ tag[0] }}</h3>
  <ul>
    {% for post in tag[1] %}
      <li><a href="{{ post.url }}">{{ post.date | date: "%B %Y" }} - {{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}