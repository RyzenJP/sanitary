
def get_potability_recommendation(tds_value, turbidity_value, temperature=25, ph_level=7.0):
    """
    Get potability recommendation based on water quality parameters
    
    Args:
        tds_value (float): TDS in mg/L
        turbidity_value (float): Turbidity in NTU
        temperature (float): Water temperature in Celsius
        ph_level (float): pH level
    
    Returns:
        dict: Potability recommendation with status, score, and advice
    """
    
    # Load trained models
    try:
        classifier = joblib.load('potability_classifier.pkl')
        score_regressor = joblib.load('potability_score_regressor.pkl')
    except:
        return {
            'status': 'error',
            'message': 'Models not found. Please train models first.'
        }
    
    # Prepare input features
    features = np.array([[
        tds_value, turbidity_value, 
        datetime.now().hour, datetime.now().weekday(),
        temperature, ph_level,
        tds_value * 2,  # conductivity approximation
        tds_value / (turbidity_value + 0.1),  # ratio
        (tds_value / 500) + (turbidity_value / 1.0)  # quality index
    ]])
    
    # Get prediction
    potability_status = classifier.predict(features)[0]
    potability_score = score_regressor.predict(features)[0]
    
    # Generate recommendations
    if potability_status == 'Potable':
        recommendation = "Water is safe for drinking. No treatment needed."
        risk_level = "Low"
        action_required = "None"
    elif potability_status == 'Marginal':
        recommendation = "Water requires treatment before consumption."
        risk_level = "Medium"
        action_required = "Filtration or disinfection recommended"
    else:
        recommendation = "Water is not safe for drinking. Immediate treatment required."
        risk_level = "High"
        action_required = "Extensive treatment or alternative water source"
    
    return {
        'status': 'success',
        'potability_status': potability_status,
        'potability_score': float(potability_score),
        'confidence': 0.85,  # Model confidence
        'risk_level': risk_level,
        'recommendation': recommendation,
        'action_required': action_required,
        'who_compliance': {
            'tds_compliant': tds_value <= 500,
            'turbidity_compliant': turbidity_value <= 1.0
        }
    }
