from django.conf import settings
from django.db import models


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
