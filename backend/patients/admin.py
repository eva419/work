from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile, Patient, PatientVitals

# Inline admin for UserProfile
class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'

# Extend the existing User admin
class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    
    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super(UserAdmin, self).get_inline_instances(request, obj)

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ['user', 'role', 'phone_number', 'created_at']
    list_filter = ['role', 'created_at']
    search_fields = ['user__username', 'user__first_name', 'user__last_name', 'phone_number']
    readonly_fields = ['created_at', 'updated_at']

@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ['patient_id', 'user', 'gender', 'age', 'registration_date']
    list_filter = ['gender', 'blood_type', 'is_active', 'registration_date']
    search_fields = ['patient_id', 'user__username', 'user__first_name', 'user__last_name']
    readonly_fields = ['patient_id', 'registration_date', 'age', 'bmi']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'patient_id', 'gender', 'registration_date')
        }),
        ('Medical Information', {
            'fields': ('blood_type', 'height', 'weight', 'medical_history', 'allergies', 'current_medications')
        }),
        ('Insurance Information', {
            'fields': ('insurance_provider', 'insurance_number')
        }),
        ('Emergency Contact', {
            'fields': ('emergency_contact_name', 'emergency_contact_phone')
        }),
        ('Status', {
            'fields': ('is_active',)
        }),
        ('Calculated Fields', {
            'fields': ('age', 'bmi'),
            'classes': ('collapse',)
        })
    )

@admin.register(PatientVitals)
class PatientVitalsAdmin(admin.ModelAdmin):
    list_display = ['patient', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'recorded_at']
    list_filter = ['recorded_at', 'recorded_by']
    search_fields = ['patient__patient_id', 'patient__user__first_name', 'patient__user__last_name']
    readonly_fields = ['recorded_at']
    
    fieldsets = (
        ('Patient Information', {
            'fields': ('patient', 'recorded_by', 'recorded_at')
        }),
        ('Vital Signs', {
            'fields': ('systolic_bp', 'diastolic_bp', 'heart_rate', 'temperature', 'respiratory_rate', 'oxygen_saturation')
        }),
        ('Measurements', {
            'fields': ('weight', 'height')
        }),
        ('Notes', {
            'fields': ('notes',)
        })
    )

