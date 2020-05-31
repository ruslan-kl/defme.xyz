---
authors:
- admin
bio: My life credo is "Never stop learning". When I am not learning, I am travelling or hiking.
email: "rklymentiev@gmail.com"
name: Ruslan Klymentiev
organizations:
- name: ""
  url: ""
role: Data Scientist
social:
- icon: envelope
  icon_pack: fas
  link: '/about/#contact'
- icon: twitter
  icon_pack: fab
  link: https://twitter.com/ruslan_kl
- icon: github
  icon_pack: fab
  link: https://github.com/ruslan-kl
superuser: true
---

<style>
.button {
  background-color: white;
  border: 2px solid red;
  color: black;
  padding: 15px 25px;
  text-align: center;
  border-radius: 14px;
  font-size: 16px;
  cursor: pointer;
  transition-duration: 0.4s;
}

.button:hover {
  background-color: red;
}
</style>

```python
class AboutMe():
    
    def __init__(self):
        self.name = 'Ruslan'
        self.last_name = 'Klymentiev'
        self.interests = []
    
    def me(self, learning):
        self.life_credo = 'Never stop learning'
        if learning:
            self.activity = 'learning'
        else:
            self.activity = 'travelling'
            
    def add_interest(self, interest):
        self.interests.append(interest)
        
    def preferences(self, r, python):
        self.love_is_equal = r == python 
    
    
RK = AboutMe()
RK.add_interest('neuroscience')
RK.add_interest('healthcare')
RK.preferences(r=True, python=True)
print(RK.love_is_equal)
>> True
```

<center>
<a class="btn" href="CV_Klymentiev.pdf" target="_blank">
<button class="button">Download my CV</button>
</a>
</center>


