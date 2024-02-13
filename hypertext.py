htext = '''
<!-- ==================== CSS ==================== -->
<style>
    .grid-container {
      display: grid;
      grid-template-columns: auto auto;
      background-color: #ffffff;
      padding: 10px;
    }
    .grid-item {
      background-color: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.8);
      padding: 30px;
      text-align: center;
      justify-content: center;
      align-items: center;
    }
    .bio {
      background-color: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.8);
      padding: 10px;
      text-align: left;
    }
    .name {
      background-color: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.8);
      padding: 10px;
      text-align: left;
    }
    .image {
      display: flex;
      background-color: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.8);
      padding: 20px;
      text-align: left; 
    }
    .image-bio {
      background-color: rgba(255, 255, 255, 0.8);
      border: 1px solid rgba(255, 255, 255, 0.8);
      padding: 20px;
      text-align: left;
    }
</style>

<!-- ==================== HTML ==================== -->

<!-- ===== ROWAN UNIVERSITY HEADER ===== -->
<div style="text-align: center;">
    <div style="padding: 20px;">
        <h1>Rowan University</h1></br> 
    </div>
    <div>
        <h1>AI Using Fractional Methods</h1>
        <h3><u>Advisors:</u> Dr. Ravi Ramachandran, Ian Nielsan</h3> 
    </div> 
</div>

<!-- ===== BIOGRAPHY GRID ===== -->
<div class="grid-container">

<!-- ===== KEIANE ===== -->
    <div class="grid-item">
        <div class="name"><h1>Keiane Balicanta</h1></div>
        <div class="image">
            <a href="https://www.linkedin.com/in/keiane-balicanta"><img src="file/keiane.png" width="200" height="150" loading="lazy" style="border-radius: 50%; filter: drop-shadow(0 0 0.5rem #313131)"></a>
            <div class="image-bio">
                <p>Department of Electrical & Computer Engineering</p>
                
            </div>
        </div>
        <div class="bio">
            <ul>
                <li>Organized code files via remote repositories</li>
                <li>Wrote custom HTML and CSS for Gradio implementation</li>
                <li>Implemented alpha slider, output plot, and RL attribution map visualization</li>
                <li>Deployed RL Visualization web application via HugginFace Spaces for site hosting</li>
            </ul>

        </div>
    </div>

<!-- ===== HENRY ===== -->
    <div class="grid-item">
        <div class="name"><h1>Henry Conde</h1></div>
        <div class="image">
            <a href="https://shorturl.at/tyM07"><img src="file/henry.jpg" width="200" height="150" loading="lazy" style="border-radius: 50%; filter: drop-shadow(0 0 0.5rem #313131)"></a>
            <div class="image-bio">
                <p>Department of Electrical & Computer Engineering</p>
            </div>
        </div>
        <div class="bio">
            <ul>
                <li>Implemented adversarial attacks</li>
                <li>Can speak Chinese</li>
                <li>Has face on $100 bill</li>
                <li>Engaged</li>
                <li>Implemented more adversarial attacks</li>
            </ul>
        </div>
    </div>

<!-- ===== ETHAN ===== -->
    <div class="grid-item">
        <div class="name"><h1>Ethan White</h1></div>
        <div class="image">
            <a href="https://shorturl.at/tyM07"><img src="file/ethan.jpg" width="200" height="150" loading="lazy" style="border-radius: 50%; filter: drop-shadow(0 0 0.5rem #313131)"></a>
            <div class="image-bio">
                <p>Department of Electrical & Computer Engineering</p>
                
            </div>
        </div>
        <div class="bio">
            <ul>
                <li>Worked on design of documentation tab of both UI interfaces</li>
                <li>Added backend checks for quality of life features (i.e., checking for CUDA, error checking parameters, pop up notifications</li>
                <li>Researched standards and design constraints for organization and planning</ul>
            </ul>   
        </div>
    </div>

<!-- ===== EVELYN ===== -->
    <div class="grid-item">
        <div class="name"><h1>Evelyn Atkins</h1></div>
        <div class="image">
            <a href="https://shorturl.at/tyM07"><img src="file/evelyn.jpg" width="200" height="150" loading="lazy" style="border-radius: 50%; filter: drop-shadow(0 0 0.5rem #313131)"></a>
            <div class="image-bio">
        <p>Department of Mechanical Engineering</p>
                
            </div>
        </div>
            <div class="bio">
            <ul>
                <li>Worked on design of documentation tab of UI interfaces</li>
                <li>Added backend checks for quality of life features (i.e., checking for CUDA, error checking parameters, pop up notifications</li>
            </ul>    
        </div>
    </div>

<!-- ===== LUKE ===== -->
    <div class="grid-item">
        <div class="name"><h1>Luke Wilkins</h1></div>
        <div class="image">
            <a href="https://shorturl.at/tyM07"><img src="file/luke.jpg" width="200" height="150" loading="lazy" style="border-radius: 50%; filter: drop-shadow(0 0 0.5rem #313131)"></a>
            <div class="image-bio">
                <p>Department of Electrical & Computer Engineering</p>
                
            </div>
        </div>
        <div class="bio">   
            <ul> 
                <li>Modified model dropdown, adversarial attack implementation, and radio settings</li>
                <li>Deployed Model Training web application via HugginFace Spaces for site hosting</li>
                <li>Lead CUDA manager</li>
                <li>Can speak French</li>
            </ul>
        </div>
    </div>

<!-- ===== MATT ===== -->
    <div class="grid-item">
        <div class="name"><h1>Matthew Gerace</h1></div>
        <div class="image">
            <a href="https://shorturl.at/tyM07"><img src="file/matt.jpg" width="200" height="150" loading="lazy" style="border-radius: 50%; filter: drop-shadow(0 0 0.5rem #313131)"></a>
            <div class="image-bio">
                <p>Department of Electrical & Computer Engineering</p>
                
            </div>
        </div>
        <div class="bio">
            <ul>
                <li>Added code optimization strategies reducing code lines for maintainability and code efficiency</li>
                <li>Took AP Spanish</li>
                <li>Can also speak Chinese</li>
                <li>Force pushed to main ???</li>
                <li>"Wo qing ni ba."</li>
            </ul>
        </div>
    </div>

</div>
'''