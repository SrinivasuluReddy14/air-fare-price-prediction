from django.db import models

class Signup(models.Model):
    username = models.CharField(max_length=100, unique=True)
    password = models.CharField(max_length=100)
    contact_no = models.CharField(max_length=15)
    email_id = models.EmailField()
    address = models.TextField()

    def __str__(self):
        return self.username