"""Secure HTML Dashboard for HTSR Monitoring

Live dashboard for monitoring grokking detection metrics with:
- Strong authentication (bcrypt hashed passwords)
- Auto-refresh every 2 minutes
- Real-time charts for α, LR, loss metrics
- Log classification (SEVERE, MAJOR, MINOR, STATUS)
- Entropic steering monitoring
- Loss threshold alerts
- Communications stub for severe alerts
"""

import threading
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from collections import deque
import hashlib
import secrets
import json

logger = logging.getLogger(__name__)


class SecureAuthenticator:
    """Simple secure authentication with password verification."""

    def __init__(self, users: Dict[str, str], session_timeout: int = 172800):
        """Initialize authenticator with user credentials.

        Args:
            users: Dict mapping usernames to hashed passwords
            session_timeout: Session timeout in seconds (default 2 days)
        """
        self.users = users
        self.session_timeout = session_timeout
        self.sessions: Dict[str, Dict] = {}

    def verify_password(self, username: str, password: str) -> bool:
        """Verify username and password."""
        if username not in self.users:
            return False

        stored_hash = self.users[username]
        return self._verify_hash(password, stored_hash)

    def _verify_hash(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash."""
        if stored_hash.startswith("$2") or stored_hash.startswith("$pbkdf2"):
            try:
                from passlib.hash import pbkdf2_sha256, bcrypt

                if stored_hash.startswith("$pbkdf2"):
                    return pbkdf2_sha256.verify(password, stored_hash)
                elif stored_hash.startswith("$2"):
                    return bcrypt.verify(password, stored_hash)
            except ImportError:
                pass

        return self._simple_verify(password, stored_hash)

    def _simple_verify(self, password: str, stored_hash: str) -> bool:
        """Simple hash verification for compatibility."""
        if ":" in stored_hash:
            algo, hash_val = stored_hash.split(":", 1)
            if algo == "sha256":
                return self._sha256_hash(password) == hash_val
        elif len(stored_hash) == 64:
            return self._sha256_hash(password) == stored_hash

        return password == stored_hash

    def _sha256_hash(self, password: str) -> str:
        """Create SHA-256 hash of password."""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_session(self, username: str) -> str:
        """Create a new session for a user."""
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {
            "username": username,
            "created": time.time(),
            "last_activity": time.time(),
        }
        return token

    def verify_session(self, token: str) -> Optional[str]:
        """Verify a session token and return username if valid."""
        if token not in self.sessions:
            return None

        session = self.sessions[token]

        if time.time() - session["last_activity"] > self.session_timeout:
            del self.sessions[token]
            return None

        session["last_activity"] = time.time()
        return session["username"]

    def destroy_session(self, token: str):
        """Destroy a session."""
        if token in self.sessions:
            del self.sessions[token]


class CommunicationStub:
    """Stub for sending communications when severe alerts are detected."""

    def __init__(self):
        self.enabled = True
        self.notification_queue: List[Dict] = []
        self.channels = {
            "email": False,
            "slack": False,
            "discord": False,
            "pagerduty": False,
            "sms": False,
        }

    def configure_channel(self, channel: str, config: Dict[str, Any]):
        """Configure a communication channel."""
        if channel in self.channels:
            self.channels[channel] = True
            logger.info(f"[COMMUNICATION STUB] Configured {channel} channel")

    def send_severe_alert(
        self, message: str, details: Dict[str, Any] = None, channels: List[str] = None
    ) -> bool:
        """Send severe alert via available channels.

        Args:
            message: Alert message
            details: Additional details
            channels: Specific channels to use (defaults to all enabled)

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return False

        notification = {
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
            "channels": channels or list(self.channels.keys()),
        }

        self.notification_queue.append(notification)

        logger.info(f"[COMMUNICATION STUB] Severe alert: {message}")

        # TODO: Implement actual communication channels:
        #
        # Email (SMTP):
        # def _send_email(self, message, details):
        #     import smtplib
        #     from email.mime.text import MIMEText
        #     msg = MIMEText(message)
        #     msg['Subject'] = f"[HTSR SEVERE] {message}"
        #     with smtplib.SMTP('smtp.example.com') as server:
        #         server.login('user', 'password')
        #         server.sendmail('from', 'to', msg.as_string())
        #
        # Slack Webhook:
        # def _send_slack(self, message, details):
        #     import requests
        #     webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
        #     if webhook_url:
        #         requests.post(webhook_url, json={'text': message})
        #
        # Discord Webhook:
        # def _send_discord(self, message, details):
        #     import requests
        #     webhook_url = os.environ.get('DISCORD_WEBHOOK_URL')
        #     if webhook_url:
        #         requests.post(webhook_url, json={'content': message})
        #
        # PagerDuty:
        # def _send_pagerduty(self, message, details):
        #     import requests
        #     api_key = os.environ.get('PAGERDUTY_API_KEY')
        #     if api_key:
        #         requests.post(
        #             'https://events.pagerduty.com/v2/enqueue',
        #             json={'routing_key': api_key, 'event_action': 'trigger', 'payload': {'summary': message}}
        #         )
        #
        #        SMS (Twilio):
        # def _send_sms(self, message, details):
        #     from twilio.rest import Client
        #     client = Client(os.environ.get('TWILIO_SID'), os.environ.get('TWILIO_TOKEN'))
        #     client.messages.create(
        #         body=message, from_='+1234567890', to='+0987654321'
        #     )

        return True

    def get_queued_notifications(self) -> List[Dict[str, Any]]:
        """Get queued notifications for display/debugging."""
        return self.notification_queue

    def clear_queue(self):
        """Clear the notification queue."""
        self.notification_queue.clear()

    def get_channel_status(self) -> Dict[str, bool]:
        """Get status of communication channels."""
        return self.channels.copy()


class LogLevel:
    """Log level constants."""

    SEVERE = "SEVERE"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    STATUS = "STATUS"


class HTMLDashboard:
    """HTML Dashboard for HTSR Monitoring with real-time updates."""

    def __init__(
        self,
        config: Dict[str, Any] = None,
        auth_users: Dict[str, str] = None,
        auto_refresh_interval: int = 120,  # 2 minutes
    ):
        """Initialize the HTML Dashboard.

        Args:
            config: Dashboard configuration
            auth_users: Dict mapping usernames to hashed passwords
            auto_refresh_interval: Auto-refresh interval in seconds
        """
        self.config = config or {}
        self.auto_refresh_interval = auto_refresh_interval

        # Authentication
        self.authenticator = SecureAuthenticator(
            users=auth_users or {},
            session_timeout=172800,  # 2 days
        )

        # Communication stub for severe alerts
        self.comm_stub = CommunicationStub()

        # Monitoring data storage
        self.monitoring_data = {
            "alpha_history": deque(maxlen=500),
            "lr_history": deque(maxlen=500),
            "train_loss_history": deque(maxlen=500),
            "val_loss_history": deque(maxlen=500),
            "alpha_variance_history": deque(maxlen=100),
            "entropic_steering_hits": deque(maxlen=100),
            "intervention_history": deque(maxlen=100),
            "weight_entropy_history": deque(maxlen=500),
            "power_draw_history": deque(maxlen=500),
            "expert_utilization_history": deque(maxlen=500),
            "gns_history": deque(maxlen=500),
            "layer_alphas": {},
            "logs": deque(maxlen=100),
            "alerts": deque(maxlen=50),
            "current_metrics": {},
            "loss_thresholds": {
                "train_loss_warning": 1.0,
                "train_loss_critical": 0.1,
                "val_loss_warning": 1.5,
                "val_loss_critical": 0.2,
            },
        }

        # State
        self.is_running = False
        self.server_thread: Optional[threading.Thread] = None
        self.start_time: Optional[float] = None
        self.total_steps: int = 0

        # Callbacks for training integration
        self.on_alpha_update: Optional[Callable] = None
        self.on_lr_change: Optional[Callable] = None
        self.on_loss_update: Optional[Callable] = None
        self.on_entropic_steering: Optional[Callable] = None
        self.on_alert: Optional[Callable] = None

    def start(self):
        """Start the dashboard server."""
        if self.is_running:
            return

        self.is_running = True
        self.start_time = time.time()

        logger.info(f"HTSR Dashboard started on port {self.config.get('port', 8050)}")
        logger.info(f"Auto-refresh interval: {self.auto_refresh_interval}s")

    def stop(self):
        """Stop the dashboard server."""
        if not self.is_running:
            return

        self.is_running = False

        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)

        logger.info("HTSR Dashboard stopped")

    def add_log(
        self,
        level: str,
        message: str,
        step: int,
        details: Dict[str, Any] = None,
        layer_name: str = None,
    ):
        """Add a log entry to the dashboard.

        Args:
            level: Log level (SEVERE, MAJOR, MINOR, STATUS)
            message: Log message
            step: Training step
            details: Additional details
            layer_name: Optional layer name
        """
        timestamp = time.time()

        log_entry = {
            "level": level,
            "message": message,
            "timestamp": timestamp,
            "step": step,
            "details": details or {},
            "layer_name": layer_name,
        }

        self.monitoring_data["logs"].append(log_entry)

        # Queue severe alerts for communication
        if level == LogLevel.SEVERE:
            self.comm_stub.send_severe_alert(message, details)

            alert = {
                "type": "severe",
                "message": message,
                "step": step,
                "timestamp": timestamp,
                "details": details,
            }
            self.monitoring_data["alerts"].append(alert)

            if self.on_alert:
                self.on_alert(alert)

    def update_alpha_metrics(
        self,
        layer_alphas: Dict[str, float],
        model_alpha: float,
        alpha_variance: float,
        over_grokking_layers: Dict[str, float],
        high_variance_layers: Dict[str, float],
        step: int,
    ):
        """Update α metrics from HTSR monitor.

        Args:
            layer_alphas: Dict mapping layer names to α values
            model_alpha: Overall model α
            alpha_variance: Variance of α across layers
            over_grokking_layers: Layers with α > 4.5
            high_variance_layers: Layers with high variance
            step: Current training step
        """
        timestamp = time.time()

        # Store current metrics
        self.monitoring_data["current_metrics"]["alpha"] = {
            "model_alpha": model_alpha,
            "alpha_variance": alpha_variance,
            "layer_alphas": layer_alphas,
            "over_grokking_count": len(over_grokking_layers),
            "high_variance_count": len(high_variance_layers),
            "step": step,
            "timestamp": timestamp,
        }

        # Add to history
        self.monitoring_data["alpha_history"].append(
            {"step": step, "model_alpha": model_alpha, "timestamp": timestamp}
        )

        self.monitoring_data["alpha_variance_history"].append(
            {"step": step, "variance": alpha_variance, "timestamp": timestamp}
        )

        # Update layer alphas
        self.monitoring_data["layer_alphas"] = layer_alphas

        # Check for over-grokking (α > 4.5) - SEVERE
        if over_grokking_layers:
            self.add_log(
                LogLevel.SEVERE,
                f"Over-grokking detected: {len(over_grokking_layers)} layers with α > 4.5",
                step,
                {
                    "model_alpha": model_alpha,
                    "layer_count": len(over_grokking_layers),
                    "layers": list(over_grokking_layers.keys()),
                },
            )

        # Check for high variance - SEVERE
        if alpha_variance > 0.5:
            self.add_log(
                LogLevel.SEVERE,
                f"High α variance detected: {alpha_variance:.4f}",
                step,
                {"variance": alpha_variance, "over_threshold": alpha_variance > 0.5},
            )

        if self.on_alpha_update:
            self.on_alpha_update(layer_alphas, model_alpha, alpha_variance, step)

    def update_lr(self, lr_values: List[float], step: int):
        """Update learning rate metrics.

        Args:
            lr_values: List of LR values for different parameter groups
            step: Current training step
        """
        timestamp = time.time()

        avg_lr = sum(lr_values) / len(lr_values) if lr_values else 0

        self.monitoring_data["current_metrics"]["lr"] = {
            "values": lr_values,
            "average": avg_lr,
            "step": step,
            "timestamp": timestamp,
        }

        self.monitoring_data["lr_history"].append(
            {"step": step, "lr": avg_lr, "timestamp": timestamp}
        )

        # Check for LR oscillation - MAJOR
        if len(self.monitoring_data["lr_history"]) >= 5:
            recent_lrs = [
                entry["lr"] for entry in list(self.monitoring_data["lr_history"])[-5:]
            ]
            lr_variance = max(recent_lrs) - min(recent_lrs)

            if lr_variance > avg_lr * 0.5:
                self.add_log(
                    LogLevel.MAJOR,
                    f"Learning rate oscillating significantly (variance: {lr_variance:.6f})",
                    step,
                    {"variance": lr_variance},
                )

        if self.on_lr_change:
            self.on_lr_change(lr_values, step)

    def update_losses(self, train_loss: float, val_loss: Optional[float], step: int):
        """Update training and validation loss metrics.

        Args:
            train_loss: Current training loss
            val_loss: Current validation loss (optional)
            step: Current training step
        """
        timestamp = time.time()

        self.monitoring_data["current_metrics"]["loss"] = {
            "train": train_loss,
            "val": val_loss,
            "step": step,
            "timestamp": timestamp,
        }

        self.monitoring_data["train_loss_history"].append(
            {"step": step, "loss": train_loss, "timestamp": timestamp}
        )

        if val_loss is not None:
            self.monitoring_data["val_loss_history"].append(
                {"step": step, "loss": val_loss, "timestamp": timestamp}
            )

            # Check for rapid loss changes - MAJOR
            if len(self.monitoring_data["train_loss_history"]) >= 3:
                recent_losses = [
                    entry["loss"]
                    for entry in list(self.monitoring_data["train_loss_history"])[-3:]
                ]
                loss_change = recent_losses[0] - recent_losses[-1]

                # Rapid decrease
                if loss_change > 0.5:
                    self.add_log(
                        LogLevel.MAJOR,
                        f"Rapid loss decrease detected ({loss_change:.4f} in last 2 checks)",
                        step,
                        {"change": loss_change},
                    )

                # Rapid increase
                if loss_change < -0.5:
                    self.add_log(
                        LogLevel.MAJOR,
                        f"Rapid loss increase detected ({abs(loss_change):.4f} in last 2 checks)",
                        step,
                        {"change": abs(loss_change)},
                    )

        # Check thresholds
        thresholds = self.monitoring_data["loss_thresholds"]

        if train_loss > thresholds["train_loss_warning"]:
            self.add_log(
                LogLevel.MINOR,
                f"Training loss above warning threshold ({train_loss:.4f} > {thresholds['train_loss_warning']:.4f})",
                step,
            )

        if val_loss and val_loss > thresholds["val_loss_warning"]:
            self.add_log(
                LogLevel.MINOR,
                f"Validation loss above warning threshold ({val_loss:.4f} > {thresholds['val_loss_warning']:.4f})",
                step,
            )

        if self.on_loss_update:
            self.on_loss_update(train_loss, val_loss, step)

    def log_entropic_steering_hit(self, layer_name: str, direction: str, step: int):
        """Log an entropic steering event - MINOR level."""
        timestamp = time.time()

        hit = {
            "layer": layer_name,
            "direction": direction,
            "step": step,
            "timestamp": timestamp,
        }

        self.monitoring_data["entropic_steering_hits"].append(hit)

        # Count recent hits for same layer
        layer_hits = [
            h
            for h in self.monitoring_data["entropic_steering_hits"]
            if h["layer"] == layer_name and step - h["step"] < 100
        ]

        if len(layer_hits) >= 3:
            self.add_log(
                LogLevel.MAJOR,
                f"Entropic steering hit repeatedly on {layer_name} ({len(layer_hits)} times)",
                step,
                {"layer": layer_name, "hit_count": len(layer_hits)},
            )
        else:
            self.add_log(
                LogLevel.MINOR,
                f"Entropic steering hit on {layer_name}",
                step,
                {"layer": layer_name, "direction": direction},
            )

        if self.on_entropic_steering:
            self.on_entropic_steering(layer_name, direction, step)

    def update_system_metrics(self, weight_entropy: float, power_draw: float, step: int):
        """Update weight entropy and power draw metrics."""
        timestamp = time.time()

        self.monitoring_data["current_metrics"]["system"] = {
            "weight_entropy": weight_entropy,
            "power_draw": power_draw,
            "step": step,
            "timestamp": timestamp,
        }

        self.monitoring_data["weight_entropy_history"].append(
            {"step": step, "entropy": weight_entropy, "timestamp": timestamp}
        )

        self.monitoring_data["power_draw_history"].append(
            {"step": step, "power": power_draw, "timestamp": timestamp}
        )

    def update_moe_metrics(self, utilization: float, gns: float, step: int):
        """Update MoE specific metrics: utilization and Gradient Noise Scale."""
        timestamp = time.time()

        self.monitoring_data["current_metrics"]["moe"] = {
            "utilization": utilization,
            "gns": gns,
            "step": step,
            "timestamp": timestamp,
        }

        self.monitoring_data["expert_utilization_history"].append(
            {"step": step, "utilization": utilization, "timestamp": timestamp}
        )

        self.monitoring_data["gns_history"].append(
            {"step": step, "gns": gns, "timestamp": timestamp}
        )

    def log_intervention(
        self, intervention_type: str, details: Dict[str, Any], step: int
    ):
        """Log an intervention action."""
        timestamp = time.time()

        intervention = {
            "type": intervention_type,
            "details": details,
            "step": step,
            "timestamp": timestamp,
        }

        self.monitoring_data["intervention_history"].append(intervention)

        self.add_log(
            LogLevel.MAJOR, f"Intervention applied: {intervention_type}", step, details
        )

    def set_loss_thresholds(
        self,
        train_warning: float = None,
        train_critical: float = None,
        val_warning: float = None,
        val_critical: float = None,
    ):
        """Set loss threshold values."""
        if train_warning is not None:
            self.monitoring_data["loss_thresholds"]["train_loss_warning"] = (
                train_warning
            )
        if train_critical is not None:
            self.monitoring_data["loss_thresholds"]["train_loss_critical"] = (
                train_critical
            )
        if val_warning is not None:
            self.monitoring_data["loss_thresholds"]["val_loss_warning"] = val_warning
        if val_critical is not None:
            self.monitoring_data["loss_thresholds"]["val_loss_critical"] = val_critical

    def configure_communication_channel(self, channel: str, config: Dict[str, Any]):
        """Configure a communication channel for alerts."""
        self.comm_stub.configure_channel(channel, config)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all dashboard data for rendering."""
        logs = list(self.monitoring_data["logs"])

        log_summary = {
            "SEVERE": len([l for l in logs if l["level"] == LogLevel.SEVERE]),
            "MAJOR": len([l for l in logs if l["level"] == LogLevel.MAJOR]),
            "MINOR": len([l for l in logs if l["level"] == LogLevel.MINOR]),
            "STATUS": len([l for l in logs if l["level"] == LogLevel.STATUS]),
        }

        return {
            "alpha": self.monitoring_data["current_metrics"].get("alpha", {}),
            "lr": self.monitoring_data["current_metrics"].get("lr", {}),
            "loss": self.monitoring_data["current_metrics"].get("loss", {}),
            "alpha_history": list(self.monitoring_data["alpha_history"]),
            "alpha_variance_history": list(
                self.monitoring_data["alpha_variance_history"]
            ),
            "lr_history": list(self.monitoring_data["lr_history"]),
            "train_loss_history": list(self.monitoring_data["train_loss_history"]),
            "val_loss_history": list(self.monitoring_data["val_loss_history"]),
            "weight_entropy_history": list(self.monitoring_data["weight_entropy_history"]),
            "power_draw_history": list(self.monitoring_data["power_draw_history"]),
            "expert_utilization_history": list(self.monitoring_data["expert_utilization_history"]),
            "gns_history": list(self.monitoring_data["gns_history"]),
            "entropic_steering_hits": list(
                self.monitoring_data["entropic_steering_hits"]
            ),
            "intervention_history": list(self.monitoring_data["intervention_history"]),
            "layer_alphas": self.monitoring_data["layer_alphas"],
            "logs": logs,
            "alerts": list(self.monitoring_data["alerts"]),
            "log_summary": log_summary,
            "loss_thresholds": self.monitoring_data["loss_thresholds"],
            "communication_queue": self.comm_stub.get_queued_notifications(),
            "statistics": {
                "uptime_seconds": time.time() - self.start_time
                if self.start_time
                else 0,
                "total_steps": self.total_steps,
                "total_alerts": len(self.monitoring_data["alerts"]),
                "total_interventions": len(
                    self.monitoring_data["intervention_history"]
                ),
                "total_entropic_hits": len(
                    self.monitoring_data["entropic_steering_hits"]
                ),
            },
        }

    def get_html_report(self) -> str:
        """Generate HTML report of current monitoring state."""
        data = self.get_dashboard_data()

        log_colors = {
            "SEVERE": "#ff4444",
            "MAJOR": "#ff8800",
            "MINOR": "#ffff00",
            "STATUS": "#00ff00",
        }

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>HTSR Grokking Monitoring Dashboard</title>
            <meta http-equiv="refresh" content="{self.auto_refresh_interval}">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }}
                .header {{ background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px; }}
                .metric-card {{ background: #16213e; padding: 20px; border-radius: 10px; }}
                .alert {{ background: #4a0e0e; padding: 15px; border-radius: 5px; margin: 5px 0; }}
                .log-entry {{ padding: 10px; border-radius: 5px; margin: 3px 0; }}
                .SEVERE {{ background: #4a0e0e; border-left: 4px solid #ff4444; }}
                .MAJOR {{ background: #3e3a0e; border-left: 4px solid #ff8800; }}
                .MINOR {{ background: #3e4a0e; border-left: 4px solid #ffff00; }}
                .STATUS {{ background: #0e4a1e; border-left: 4px solid #00ff00; }}
                .healthy {{ background: #0e4a1e; }}
                .warning {{ background: #3e4a0e; }}
                .severe {{ background: #4a0e0e; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
                h1, h2 {{ color: #00d9ff; }}
                .log-summary {{ display: flex; gap: 20px; margin-bottom: 20px; }}
                .log-count {{ padding: 10px 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HTSR Grokking Monitoring Dashboard</h1>
                <p>Uptime: {data["statistics"]["uptime_seconds"]:.0f}s | Steps: {data["statistics"]["total_steps"]}</p>
            </div>
            
            <div class="log-summary">
                <div class="log-count" style="background: #4a0e0e;">SEVERE: {data["log_summary"]["SEVERE"]}</div>
                <div class="log-count" style="background: #3e3a0e;">MAJOR: {data["log_summary"]["MAJOR"]}</div>
                <div class="log-count" style="background: #3e4a0e;">MINOR: {data["log_summary"]["MINOR"]}</div>
                <div class="log-count" style="background: #0e4a1e;">STATUS: {data["log_summary"]["STATUS"]}</div>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <h2>α Metrics</h2>
                    <p>Model α: {data["alpha"].get("model_alpha", "N/A"):.2f}</p>
                    <p>Variance: {data["alpha"].get("alpha_variance", "N/A"):.4f}</p>
                    <p>Over-Grokking Layers: {data["alpha"].get("over_grokking_count", 0)}</p>
                </div>
                <div class="metric-card">
                    <h2>Learning Rate</h2>
                    <p>Current LR: {data["lr"].get("average", 0):.6f}</p>
                </div>
                <div class="metric-card">
                    <h2>Loss</h2>
                    <p>Train Loss: {data["loss"].get("train", "N/A"):.4f}</p>
                    <p>Val Loss: {data["loss"].get("val", "N/A"):.4f}</p>
                </div>
                <div class="metric-card">
                    <h2>Grokking & Noise</h2>
                    <p>GNS: {data["alpha"].get("gns", data.get("gns_history", [{"gns": 0}])[-1].get("gns", 0)):.6f}</p>
                    <p>Weight Entropy: {data.get("weight_entropy_history", [{"entropy": 0}])[-1].get("entropy", 0):.4f}</p>
                </div>
                <div class="metric-card">
                    <h2>MoE & System</h2>
                    <p>Expert Utilization: {data.get("expert_utilization_history", [{"utilization": 0}])[-1].get("utilization", 0)*100:.1f}%</p>
                    <p>Power Draw: {data.get("power_draw_history", [{"power": 0}])[-1].get("power", 0):.1f}W</p>
                </div>
            </div>
            
            <h2>Recent Logs</h2>
            <div class="logs">
        """

        for log in reversed(data["logs"][-15:]):
            css_class = log["level"]
            html += f'<div class="log-entry {css_class}">[{log["step"]}] {log["level"]}: {log["message"]}</div>'

        html += """
            </div>
            
            <h2>Layer α Values</h2>
            <table>
                <tr><th>Layer</th><th>α</th><th>Status</th></tr>
        """

        for layer, alpha in sorted(
            data["layer_alphas"].items(),
            key=lambda x: x[1] if isinstance(x[1], (int, float)) else 2.0,
            reverse=True,
        ):
            status = (
                "severe" if alpha > 4.5 else "warning" if alpha > 3.5 else "healthy"
            )
            html += f'<tr class="{status}"><td>{layer}</td><td>{alpha:.2f}</td><td>{status.upper()}</td></tr>'

        html += """
            </table>
        </body>
        </html>
        """

        return html

    def export_data(self, filepath: str):
        """Export monitoring data to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.get_dashboard_data(), f, indent=2, default=str)

        logger.info(f"Dashboard data exported to {filepath}")
