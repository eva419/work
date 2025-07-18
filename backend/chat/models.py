from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator
import uuid

def chat_attachment_path(instance, filename):
    """Generate upload path for chat attachments"""
    return f"chat_attachments/{instance.conversation.id}/{filename}"

class ChatConversation(models.Model):
    """Chat conversation between patients and doctors"""
    CONVERSATION_TYPE_CHOICES = [
        ('consultation', 'Medical Consultation'),
        ('follow_up', 'Follow-up'),
        ('emergency', 'Emergency'),
        ('general', 'General Inquiry'),
        ('support', 'Technical Support'),
    ]
    
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('closed', 'Closed'),
        ('archived', 'Archived'),
        ('blocked', 'Blocked'),
    ]
    
    conversation_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    participants = models.ManyToManyField(User, related_name='chat_conversations')
    
    # Conversation details
    title = models.CharField(max_length=200, blank=True)
    conversation_type = models.CharField(max_length=20, choices=CONVERSATION_TYPE_CHOICES, default='general')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Participants (specific roles)
    patient = models.ForeignKey(
        'patients.Patient', 
        on_delete=models.CASCADE, 
        related_name='chat_conversations',
        null=True, 
        blank=True
    )
    doctor = models.ForeignKey(
        'doctors.Doctor', 
        on_delete=models.CASCADE, 
        related_name='chat_conversations',
        null=True, 
        blank=True
    )
    
    # Conversation metadata
    started_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='started_conversations')
    is_emergency = models.BooleanField(default=False)
    priority = models.CharField(max_length=10, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('urgent', 'Urgent'),
    ], default='medium')
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_message_at = models.DateTimeField(null=True, blank=True)
    closed_at = models.DateTimeField(null=True, blank=True)
    
    # Settings
    is_encrypted = models.BooleanField(default=True)
    auto_archive_days = models.IntegerField(default=30)
    
    def __str__(self):
        participants = ", ".join([user.get_full_name() for user in self.participants.all()[:2]])
        return f"Conversation: {participants}"
    
    @property
    def unread_count(self):
        """Get total unread messages in conversation"""
        return self.messages.filter(is_read=False).count()
    
    @property
    def last_message(self):
        """Get the last message in conversation"""
        return self.messages.order_by('-timestamp').first()
    
    class Meta:
        verbose_name = "Chat Conversation"
        verbose_name_plural = "Chat Conversations"
        ordering = ['-updated_at']

class ChatMessage(models.Model):
    """Individual chat messages"""
    MESSAGE_TYPE_CHOICES = [
        ('text', 'Text Message'),
        ('image', 'Image'),
        ('file', 'File'),
        ('voice', 'Voice Message'),
        ('system', 'System Message'),
        ('prescription', 'Prescription'),
        ('appointment', 'Appointment'),
    ]
    
    conversation = models.ForeignKey(ChatConversation, on_delete=models.CASCADE, related_name='messages')
    sender = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sent_chat_messages')
    
    # Message content
    message_type = models.CharField(max_length=20, choices=MESSAGE_TYPE_CHOICES, default='text')
    content = models.TextField(blank=True)
    
    # Message metadata
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    read_by = models.ManyToManyField(User, blank=True, related_name='read_messages')
    
    # Message status
    is_edited = models.BooleanField(default=False)
    edited_at = models.DateTimeField(null=True, blank=True)
    is_deleted = models.BooleanField(default=False)
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    # Reply functionality
    reply_to = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True, related_name='replies')
    
    # Timestamps
    timestamp = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.sender.get_full_name()}: {self.content[:50]}..."
    
    def mark_as_read(self, user):
        """Mark message as read by a specific user"""
        if not self.read_by.filter(id=user.id).exists():
            self.read_by.add(user)
            if not self.is_read:
                self.is_read = True
                self.read_at = timezone.now()
                self.save()
    
    class Meta:
        verbose_name = "Chat Message"
        verbose_name_plural = "Chat Messages"
        ordering = ['timestamp']

class ChatAttachment(models.Model):
    """File attachments in chat messages"""
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, related_name='attachments')
    file = models.FileField(
        upload_to=chat_attachment_path,
        validators=[FileExtensionValidator(allowed_extensions=[
            'pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png', 'gif', 
            'mp3', 'wav', 'mp4', 'avi', 'txt', 'csv', 'xlsx'
        ])]
    )
    
    # File metadata
    original_filename = models.CharField(max_length=255)
    file_size = models.IntegerField(help_text="File size in bytes")
    file_type = models.CharField(max_length=10)
    mime_type = models.CharField(max_length=100, blank=True)
    
    # Image/video specific
    thumbnail = models.ImageField(upload_to='chat_thumbnails/', null=True, blank=True)
    duration = models.FloatField(null=True, blank=True, help_text="Duration in seconds for audio/video")
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Attachment: {self.original_filename}"
    
    @property
    def file_size_mb(self):
        """Return file size in MB"""
        return round(self.file_size / (1024 * 1024), 2)
    
    class Meta:
        verbose_name = "Chat Attachment"
        verbose_name_plural = "Chat Attachments"
        ordering = ['-uploaded_at']

class ChatNotification(models.Model):
    """Notifications for chat events"""
    NOTIFICATION_TYPE_CHOICES = [
        ('new_message', 'New Message'),
        ('new_conversation', 'New Conversation'),
        ('conversation_closed', 'Conversation Closed'),
        ('emergency_message', 'Emergency Message'),
        ('mention', 'Mention'),
        ('file_shared', 'File Shared'),
    ]
    
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_notifications')
    conversation = models.ForeignKey(ChatConversation, on_delete=models.CASCADE, related_name='notifications')
    message = models.ForeignKey(ChatMessage, on_delete=models.CASCADE, null=True, blank=True)
    
    notification_type = models.CharField(max_length=20, choices=NOTIFICATION_TYPE_CHOICES)
    title = models.CharField(max_length=200)
    content = models.TextField(blank=True)
    
    # Status
    is_read = models.BooleanField(default=False)
    is_sent = models.BooleanField(default=False)
    sent_via_email = models.BooleanField(default=False)
    sent_via_sms = models.BooleanField(default=False)
    sent_via_push = models.BooleanField(default=False)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    read_at = models.DateTimeField(null=True, blank=True)
    sent_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Notification for {self.recipient.username}: {self.title}"
    
    class Meta:
        verbose_name = "Chat Notification"
        verbose_name_plural = "Chat Notifications"
        ordering = ['-created_at']

class ChatTemplate(models.Model):
    """Pre-defined message templates for common responses"""
    TEMPLATE_CATEGORY_CHOICES = [
        ('greeting', 'Greeting'),
        ('appointment', 'Appointment'),
        ('prescription', 'Prescription'),
        ('follow_up', 'Follow-up'),
        ('emergency', 'Emergency'),
        ('general', 'General'),
        ('closing', 'Closing'),
    ]
    
    title = models.CharField(max_length=200)
    category = models.CharField(max_length=20, choices=TEMPLATE_CATEGORY_CHOICES)
    content = models.TextField()
    
    # Usage tracking
    usage_count = models.IntegerField(default=0)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)
    
    # Availability
    is_public = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.title} ({self.category})"
    
    class Meta:
        verbose_name = "Chat Template"
        verbose_name_plural = "Chat Templates"
        ordering = ['category', 'title']

class ChatSettings(models.Model):
    """User-specific chat settings"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='chat_settings')
    
    # Notification preferences
    email_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)
    push_notifications = models.BooleanField(default=True)
    sound_notifications = models.BooleanField(default=True)
    
    # Chat preferences
    auto_archive_conversations = models.BooleanField(default=True)
    show_online_status = models.BooleanField(default=True)
    allow_file_sharing = models.BooleanField(default=True)
    max_file_size_mb = models.IntegerField(default=10)
    
    # Privacy settings
    allow_emergency_contact = models.BooleanField(default=True)
    block_unknown_users = models.BooleanField(default=False)
    
    # Appearance
    theme = models.CharField(max_length=20, choices=[
        ('light', 'Light'),
        ('dark', 'Dark'),
        ('auto', 'Auto'),
    ], default='light')
    
    font_size = models.CharField(max_length=10, choices=[
        ('small', 'Small'),
        ('medium', 'Medium'),
        ('large', 'Large'),
    ], default='medium')
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Chat settings for {self.user.username}"
    
    class Meta:
        verbose_name = "Chat Settings"
        verbose_name_plural = "Chat Settings"

class OnlineStatus(models.Model):
    """Track user online status for chat"""
    STATUS_CHOICES = [
        ('online', 'Online'),
        ('away', 'Away'),
        ('busy', 'Busy'),
        ('offline', 'Offline'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='online_status')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='offline')
    last_seen = models.DateTimeField(auto_now=True)
    is_typing = models.BooleanField(default=False)
    typing_in_conversation = models.ForeignKey(
        ChatConversation, 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True
    )
    
    def __str__(self):
        return f"{self.user.username} - {self.status}"
    
    class Meta:
        verbose_name = "Online Status"
        verbose_name_plural = "Online Status"

