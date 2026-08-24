def generate_recommendations(inputs: dict, predicted_quality: str) -> dict:
    
    reasons = []
    tips = []

    sleep_duration = float(inputs.get('sleep_duration', 7.0))
    screen_time = int(inputs.get('screen_time', 0))
    stress_level = int(inputs.get('stress_level', 0))
    exercise_duration = int(inputs.get('exercise_duration', 0))
    caffeine = str(inputs.get('caffeine_intake', 'None'))
    interruptions = str(inputs.get('sleep_interruptions', 'No'))
    mood = str(inputs.get('mood', 'Neutral'))
    bedtime_str = str(inputs.get('bedtime', '23:00'))

    try:
        if ':' in bedtime_str:
            bedtime_hour = float(bedtime_str.split(':')[0]) + float(bedtime_str.split(':')[1]) / 60.0
        else:
            bedtime_hour = float(bedtime_str)
    except Exception:
        bedtime_hour = 23.0

    if screen_time >= 30:
        reasons.append(f"High screen time before bed ({screen_time} mins) delays natural melatonin production.")
        tips.append("📱 **Reduce Screen Usage**: Turn off digital screens (phones, TVs, laptops) 45-60 minutes before bedtime or activate night light mode.")

    if stress_level >= 6:
        reasons.append(f"Elevated stress level ({stress_level}/10) stimulates cortisol release and prevents deep relaxation.")
        tips.append("🧘 **Manage Stress Before Bed**: Practice 10 minutes of deep breathing exercises (4-7-8 method), light stretching, or mindfulness meditation.")

    if sleep_duration < 7.0:
        reasons.append(f"Short sleep duration ({sleep_duration:.1f} hrs) is below the recommended 7–9 hour target for adults.")
        tips.append("⏰ **Prioritize Sleep Time**: Adjust your daily schedule to allow for at least 7 to 8 hours of continuous, restorative sleep.")
    elif sleep_duration > 9.5:
        reasons.append(f"Excessive sleep duration ({sleep_duration:.1f} hrs) can lead to sleep inertia and daytime grogginess.")
        tips.append("⌛ **Regulate Sleep Window**: Keep your total sleep duration closer to 7.5–8.5 hours to avoid feeling sluggish.")

    if exercise_duration < 30:
        reasons.append(f"Low physical activity level ({exercise_duration} mins/day).")
        tips.append("🏃 **Increase Daily Movement**: Aim for at least 30 minutes of moderate physical activity daily (preferably in the morning or afternoon).")

    if caffeine in ['Moderate', 'High']:
        reasons.append(f"Significant caffeine consumption ('{caffeine}') disrupts sleep architecture and delays sleep onset.")
        tips.append("☕ **Limit Afternoon Caffeine**: Avoid caffeinated beverages (coffee, tea, energy drinks) after 2:00 PM or within 6 hours of bedtime.")

    if interruptions == 'Yes':
        reasons.append("Frequent sleep interruptions interrupt critical REM and deep sleep stages.")
        tips.append("🌙 **Optimize Sleep Environment**: Ensure your bedroom is pitch black, quiet, and cool (~65°F / 18°C). Limit fluids 2 hours before sleep.")

    if bedtime_hour >= 23.0 or bedtime_hour < 4.0:
        reasons.append("Late bedtime schedule desynchronizes circadian rhythms.")
        tips.append("😴 **Maintain Consistent Schedule**: Go to bed before 11:00 PM and wake up at the same time every day, including weekends.")

    if mood in ['Anxious', 'Sad']:
        reasons.append(f"Negative pre-sleep emotional state ('{mood}').")
        tips.append("📖 **Pre-sleep Wind-down Routine**: Spend 15 minutes journaling, reading a physical book, or listening to calm acoustic music before sleeping.")

    if predicted_quality == 'Good' and len(reasons) <= 1:
        reasons.append("Great sleep habits! Your lifestyle factors support healthy sleep architecture.")
        tips.append("✨ **Keep It Up**: Maintain your consistent bedtime, low stress levels, and regular exercise routine.")
    elif len(tips) == 0:
        tips.append("🌟 **General Wellness**: Maintain a dark, quiet bedroom, stick to a fixed schedule, and stay hydrated throughout the day.")

    return {
        'reasons': reasons,
        'tips': tips
    }
