from django.conf import settings
from django.db import models


class UserSettings(models.Model):
    THEME_CHOICES = [
        ("dark", "Dark"),
        ("light", "Light"),
        ("auto", "Auto"),
    ]

    LANGUAGE_CHOICES = [
        ("en", "English"),
        ("ar", "Arabic"),
        ("auto", "Auto"),
    ]

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="settings"
    )

    # Profile
    display_name = models.CharField(max_length=100, blank=True, default="")
    avatar_url = models.URLField(blank=True, default="")

    # Theme
    theme = models.CharField(max_length=10, choices=THEME_CHOICES, default="dark")
    language = models.CharField(max_length=10, choices=LANGUAGE_CHOICES, default="en")

    # Chat
    default_agent = models.CharField(max_length=50, default="chat")
    show_timestamps = models.BooleanField(default=True)
    send_on_enter = models.BooleanField(default=True)
    font_size = models.IntegerField(default=14)

    # Notifications
    email_notifications = models.BooleanField(default=True)
    training_notifications = models.BooleanField(default=True)
    sound_enabled = models.BooleanField(default=True)

    # Training
    default_model = models.CharField(max_length=255, default="google/gemma-2-2b-it")
    default_dataset_size = models.CharField(max_length=10, default="100k")
    default_epochs = models.IntegerField(default=3)
    default_learning_rate = models.FloatField(default=2e-4)

    # API Keys (encrypted in production)
    openai_api_key = models.CharField(max_length=255, blank=True, default="")
    huggingface_token = models.CharField(max_length=255, blank=True, default="")
    smtp_user = models.CharField(max_length=255, blank=True, default="")
    smtp_pass = models.CharField(max_length=255, blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "User Settings"
        verbose_name_plural = "User Settings"

    def __str__(self):
        return f"Settings for {self.user.username}"

    @classmethod
    def get_or_create(cls, user):
        """Get or create settings for a user."""
        settings_obj, created = cls.objects.get_or_create(user=user)
        return settings_obj


class TrainingJob(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        null=True, blank=True, related_name="training_jobs"
    )
    model_name = models.CharField(max_length=255, default="google/gemma-2-2b-it")
    dataset_size = models.CharField(max_length=10, default="100k")
    total_samples = models.IntegerField(default=100000)
    batch_size = models.IntegerField(default=10000)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    current_round = models.IntegerField(default=0)
    total_rounds = models.IntegerField(default=10)
    current_loss = models.FloatField(null=True, blank=True)
    email = models.EmailField(help_text="Email for notifications")
    notify_on_complete = models.BooleanField(default=True)
    error_message = models.TextField(blank=True, null=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Training {self.model_name} ({self.status})"

    def progress_percent(self):
        if self.total_rounds == 0:
            return 0
        return int((self.current_round / self.total_rounds) * 100)


class Conversation(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        null=True, blank=True, related_name="conversations"
    )
    agent_name = models.CharField(max_length=100)
    title = models.CharField(max_length=255, blank=True, null=True, help_text="Custom display name")
    is_pinned = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-is_pinned", "-updated_at"]

    def display_name(self) -> str:
        if self.title:
            return self.title
        first_msg = self.messages.filter(role="user").first()
        if first_msg:
            preview = first_msg.content[:50]
            return f"{self.agent_name}: {preview}"
        return f"{self.agent_name} #{self.id}"

    def __str__(self):
        return self.display_name()


class Message(models.Model):
    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
        ("system", "System"),
        ("tool", "Tool"),
    ]
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"[{self.role}] {self.content[:60]}..."


class Rule(models.Model):
    RULE_TYPES = [
        ("respond", "Custom Response"),
        ("transform", "Transform Input"),
    ]
    trigger = models.CharField(max_length=255, help_text="Keyword or phrase to trigger the rule")
    action = models.TextField(help_text="Response text or transformation template")
    rule_type = models.CharField(max_length=20, choices=RULE_TYPES, default="respond")
    agent_key = models.CharField(max_length=50, blank=True, null=True, help_text="Optional: restrict to specific agent")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"[{self.rule_type}] '{self.trigger}' → '{self.action[:40]}'"


class UploadedFile(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="uploaded_files"
    )
    file = models.FileField(upload_to="uploads/%Y/%m/%d/")
    filename = models.CharField(max_length=255)
    file_type = models.CharField(max_length=50, blank=True)
    content_text = models.TextField(blank=True, help_text="Extracted text content")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.filename
