---
authors:
- admin
bio: My life credo is "Never stop learning". When I am not learning, I am travelling or hiking.
email: "rklymentiev@gmail.com"
name: Ruslan Klymentiev
organizations:
- name: ""
  url: ""
role: 50% Data Scientist, 50% Neuroscientist
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
class AboutMe:
    
    def __init__(self, name=None):
        self.name = name
        self.interests = []
    
    def me(self, life_credo, learning=True):
        self.life_credo = life_credo
        if learning:
            self.activity = 'learning'
        else:
            self.activity = 'travelling'
            
    def add_interest(self, interest):
        if interest not in self.interests:
            self.interests.append(interest)
        
    def preferences(self, r, python):
        self.love_is_equal = r == python 
    
    
RK = AboutMe(name='Ruslan Klymentiev')
RK.me(life_credo='Never stop learning')
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


