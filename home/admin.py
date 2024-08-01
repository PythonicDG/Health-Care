from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from . models import CustomUser, Survey

# Register your models here.

class CustomUserInline(admin.StackedInline):
    model = CustomUser
    can_delete = False
    verbose_name_plural = 'CustomUserss'



class CustomizedUserAdmin(UserAdmin):
    inlines=(CustomUserInline,)

admin.site.unregister(User)
admin.site.register(User,CustomizedUserAdmin)

admin.site.register(CustomUser)

admin.site.register(Survey)