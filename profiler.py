import pandas as pd
import numpy as np


# ── Segment Naming ────────────────────────────────────────────────────────────
CRUISE_SEGMENT_NAMES = {
    "high_value_loyal": "Platinum Loyalists",
    "high_value_new": "Premium Explorers",
    "mid_value_regular": "Steady Voyagers",
    "low_value_inactive": "Dormant Passengers",
    "at_risk": "At-Risk Members",
    "budget_frequent": "Budget Adventurers",
    "default": "Segment"
}

SEGMENT_COLORS = [
    "#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
    "#3B1F2B", "#44BBA4", "#E94F37", "#393E41"
]

RECOMMENDATIONS = {
    "Platinum Loyalists": {
        "icon": "👑",
        "description": "High-value, long-term customers with excellent engagement.",
        "actions": [
            "Offer exclusive suite upgrades and priority boarding",
            "Invite to VIP loyalty events and pre-launch sailings",
            "Provide dedicated concierge service",
            "Early access to new route announcements",
            "Personalised anniversary/milestone rewards"
        ],
        "priority": "Retain & Reward"
    },
    "Premium Explorers": {
        "icon": "🌟",
        "description": "High spenders who are relatively new — strong upgrade potential.",
        "actions": [
            "Accelerate loyalty tier progression with bonus points",
            "Offer membership upgrade incentives",
            "Target with premium cabin promotions",
            "Assign personal travel advisor",
            "Send curated itinerary recommendations"
        ],
        "priority": "Convert to Loyalists"
    },
    "Steady Voyagers": {
        "icon": "⚓",
        "description": "Regular customers with moderate spend — the backbone segment.",
        "actions": [
            "Offer group booking discounts to increase trip frequency",
            "Cross-sell onboard packages (spa, dining, excursions)",
            "Introduce referral programme with travel credits",
            "Seasonal promotions aligned to booking history",
            "Nudge towards next loyalty tier with progress tracker"
        ],
        "priority": "Grow & Upsell"
    },
    "Budget Adventurers": {
        "icon": "🎒",
        "description": "Frequent travelers but low spend — price-sensitive segment.",
        "actions": [
            "Target with early-bird and flash sale pricing",
            "Offer value-added bundles (drink package + dining)",
            "Promote shorter itineraries and port-intensive routes",
            "Gamify loyalty with points for onboard spend",
            "Introduce installment payment options"
        ],
        "priority": "Increase Spend per Trip"
    },
    "At-Risk Members": {
        "icon": "⚠️",
        "description": "Previously engaged customers showing signs of disengagement.",
        "actions": [
            "Immediate win-back campaign with personalised offer",
            "Conduct satisfaction survey to identify pain points",
            "Offer complimentary upgrade on next booking",
            "Assign dedicated retention specialist",
            "Re-engage with exclusive member-only rates"
        ],
        "priority": "Urgent Retention"
    },
    "Dormant Passengers": {
        "icon": "💤",
        "description": "Low engagement, infrequent trips, long since last booking.",
        "actions": [
            "Re-activation email series with strong incentive",
            "Offer significant first-rebook discount",
            "Showcase new destinations and ship upgrades",
            "Simplify rebooking with one-click offers",
            "Consider removal from active marketing if unresponsive"
        ],
        "priority": "Re-activate or Sunset"
    }
}


def name_segment(segment_profile: dict) -> str:
    """Auto-name a segment based on its characteristics."""
    trips = segment_profile.get("total_trips", 0)
    spend = segment_profile.get("avg_spend_per_trip", 0)
    loyalty = segment_profile.get("loyalty_years", 0)
    days_since = segment_profile.get("days_since_last_trip", 0)
    complaints = segment_profile.get("num_complaints", 0)

    if loyalty > 10 and spend > 7000 and trips > 15:
        return "Platinum Loyalists"
    elif spend > 6000 and loyalty < 5:
        return "Premium Explorers"
    elif days_since > 60 and complaints > 1:
        return "At-Risk Members"
    elif days_since > 80:
        return "Dormant Passengers"
    elif trips > 10 and spend < 3000:
        return "Budget Adventurers"
    else:
        return "Steady Voyagers"


def profile_segments(original_df, clean_df, labels, feature_names):
    """Generate full profile for each segment."""
    df = original_df.copy()
    df["cluster"] = labels
    df["cluster_label"] = labels

    profiles = []
    unique_clusters = sorted([c for c in set(labels) if c != -1])

    for cluster_id in unique_clusters:
        cluster_df = df[df["cluster"] == cluster_id]
        cluster_clean = clean_df[labels == cluster_id]

        # Calculate means for numeric columns
        numeric_cols = cluster_clean.select_dtypes(include=[np.number]).columns
        means = cluster_clean[numeric_cols].mean().to_dict()

        # Auto-name segment
        segment_name = name_segment(means)

        # Get recommendation
        rec = RECOMMENDATIONS.get(segment_name, {
            "icon": "📊",
            "description": f"Segment {cluster_id} — review characteristics for custom strategy.",
            "actions": ["Analyse further to develop targeted strategy"],
            "priority": "Review"
        })

        # Size
        size = len(cluster_df)
        pct = round(size / len(df) * 100, 1)

        profiles.append({
            "cluster_id": cluster_id,
            "segment_name": segment_name,
            "size": size,
            "percentage": pct,
            "color": SEGMENT_COLORS[cluster_id % len(SEGMENT_COLORS)],
            "means": means,
            "icon": rec["icon"],
            "description": rec["description"],
            "actions": rec["actions"],
            "priority": rec["priority"],
            "df": cluster_df
        })

    return profiles


def get_segment_comparison_df(profiles, feature_names):
    """Build a comparison DataFrame across all segments."""
    rows = []
    for p in profiles:
        row = {
            "Segment": f"{p['icon']} {p['segment_name']}",
            "Size": p["size"],
            "% of Total": f"{p['percentage']}%",
            "Priority": p["priority"]
        }
        for feat in feature_names:
            if feat in p["means"]:
                row[feat.replace("_", " ").title()] = round(p["means"][feat], 1)
        rows.append(row)
    return pd.DataFrame(rows)