"""
MercurioAI Event Bus Module

This module implements an event-driven architecture for MercurioAI,
allowing components to communicate via events rather than direct coupling.
"""
import asyncio
import logging
from typing import Dict, List, Any, Callable, Awaitable, Optional
from collections import defaultdict
import uuid
import time

logger = logging.getLogger(__name__)

EventHandler = Callable[[Dict[str, Any]], Awaitable[None]]

class EventBus:
    """
    Central event bus that manages publishing and subscribing to events.
    Implements the Observer pattern for decoupled communication between components.
    """
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern for EventBus"""
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.subscribers = defaultdict(list)
            cls._instance.history = {}  # Store recent events for diagnostics
            cls._instance.max_history = 100  # Maximum events to store
            logger.info("EventBus initialized")
        return cls._instance
    
    async def publish(self, event_type: str, data: Dict[str, Any], retain: bool = False) -> str:
        """
        Publish an event to all subscribers
        
        Args:
            event_type: Type of event (e.g., 'market_data_updated', 'trade_executed')
            data: Event payload as dictionary
            retain: Whether to retain this event for late subscribers
            
        Returns:
            Event ID (UUID)
        """
        event_id = str(uuid.uuid4())
        timestamp = time.time()
        
        event = {
            "id": event_id,
            "type": event_type,
            "timestamp": timestamp,
            "data": data
        }
        
        # Store in history (with limited size)
        self.history[event_id] = event
        if len(self.history) > self.max_history:
            # Remove oldest events
            oldest = sorted(self.history.items(), key=lambda x: x[1]["timestamp"])[:len(self.history) - self.max_history]
            for k, _ in oldest:
                del self.history[k]
        
        # Notify subscribers
        tasks = []
        for handler, filter_func in self.subscribers[event_type]:
            # Check if this subscriber should receive this event
            if filter_func is None or filter_func(data):
                tasks.append(asyncio.create_task(self._notify_subscriber(handler, event)))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Event {event_type} ({event_id}) published to {len(tasks)} subscribers")
        else:
            logger.debug(f"Event {event_type} ({event_id}) published but no subscribers")
            
        return event_id
    
    async def _notify_subscriber(self, handler: EventHandler, event: Dict[str, Any]) -> None:
        """
        Notify a single subscriber of an event, with error handling
        
        Args:
            handler: Subscriber's callback function
            event: Event data
        """
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in event handler: {e}")
    
    def subscribe(self, 
                  event_type: str, 
                  handler: EventHandler, 
                  filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None) -> str:
        """
        Subscribe to events of a specific type
        
        Args:
            event_type: Type of event to subscribe to
            handler: Async callback function to be called when event occurs
            filter_func: Optional function to filter events
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        self.subscribers[event_type].append((handler, filter_func))
        logger.debug(f"New subscription to {event_type} events (ID: {subscription_id})")
        return subscription_id
    
    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """
        Unsubscribe from an event type
        
        Args:
            event_type: Type of event
            handler: Handler to remove
            
        Returns:
            True if unsubscribed successfully, False otherwise
        """
        for i, (h, _) in enumerate(self.subscribers[event_type]):
            if h == handler:
                self.subscribers[event_type].pop(i)
                logger.debug(f"Unsubscribed handler from {event_type} events")
                return True
        return False
    
    def get_recent_events(self, event_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent events, optionally filtered by type
        
        Args:
            event_type: Optional event type to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of recent events, newest first
        """
        events = list(self.history.values())
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        
        # Sort by timestamp (newest first) and limit
        return sorted(events, key=lambda x: x["timestamp"], reverse=True)[:limit]


# Global event types
class EventType:
    """Constants for common event types"""
    # Market data events
    MARKET_DATA_UPDATED = "market_data_updated"
    PRICE_ALERT = "price_alert"
    VOLUME_SPIKE = "volume_spike"
    
    # Trading events
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_REJECTED = "order_rejected"
    TRADE_EXECUTED = "trade_executed"
    
    # Strategy events
    SIGNAL_GENERATED = "signal_generated"
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_STOPPED = "strategy_stopped"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_OCCURRED = "error_occurred"
    
    # Backtest events
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"
    BACKTEST_PROGRESS = "backtest_progress"


# Example usage
"""
# Subscribe to events
async def handle_market_data(event):
    print(f"Market data updated: {event['data']}")

event_bus = EventBus()
event_bus.subscribe(EventType.MARKET_DATA_UPDATED, handle_market_data)

# Publish an event
await event_bus.publish(
    EventType.MARKET_DATA_UPDATED, 
    {"symbol": "AAPL", "price": 150.25, "timestamp": "2023-01-01T12:00:00Z"}
)
"""
