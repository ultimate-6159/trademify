"""Test OmniscientIntelligence"""
from trading.omniscient_intelligence import OmniscientIntelligence
import numpy as np

omni = OmniscientIntelligence()

# Test with sample data
prices = np.array([100 + i*0.1 for i in range(100)], dtype=np.float32)
highs = prices * 1.01
lows = prices * 0.99
volumes = np.ones(100, dtype=np.float32) * 1000

try:
    result = omni.analyze(
        symbol='XAUUSDm',
        signal_side='BUY',
        current_price=float(prices[-1]),
        prices=prices,
        highs=highs,
        lows=lows,
        volumes=volumes,
        atr=0.5,
        base_confidence=75,  # ????????? 75
        balance=10000,
        equity=10000
    )
    
    print('='*80)
    print('?? OMNISCIENT INTELLIGENCE ANALYSIS RESULT')
    print('='*80)
    print(f'\n? CAN TRADE: {result.can_trade}')
    print(f'?? OMNISCIENT SCORE: {result.omniscient_score:.2f}/100')
    print(f'?? CONSCIOUSNESS: {result.consciousness_level.value}')
    print(f'?? UNIVERSAL ALIGNMENT: {result.universal_alignment:.2f}%')
    print(f'?? WIN PROBABILITY: {result.win_probability:.1%}')
    print(f'?? CONFIDENCE: {result.confidence:.2f}')
    print(f'?? EDGE: {result.edge:.2f}%')
    print(f'?? POSITION SIZE: {result.omniscient_position_size:.4f}')
    
    print(f'\n?? ENTRY TARGETS:')
    print(f'  Entry: {result.optimal_entry:.5f}')
    print(f'  SL: {result.optimal_sl:.5f}')
    print(f'  TP: {result.optimal_tp:.5f}')
    print(f'  R:R: {result.expected_rr:.2f}')
    
    print(f'\n? PHYSICS STATE: {result.physics.physics_state.value}')
    print(f'  Velocity: {result.physics.price_velocity:.4f}')
    print(f'  Acceleration: {result.physics.price_acceleration:.4f}')
    print(f'  Momentum Energy: {result.physics.momentum_energy:.4f}')
    
    print(f'\n?? NEURAL ENSEMBLE: {result.neural.confidence.value}')
    print(f'  Vote: {result.neural.ensemble_vote}')
    print(f'  LSTM Prediction: {result.neural.lstm_prediction:.2f}')
    print(f'  Pattern Score: {result.neural.cnn_pattern_score:.2f}')
    print(f'  Uncertainty: {result.neural.uncertainty:.3f}')
    
    print(f'\n?? CHAOS METRICS: {result.chaos.chaos_level.value}')
    print(f'  Lyapunov: {result.chaos.lyapunov_exponent:.4f}')
    print(f'  Fractal Dim: {result.chaos.fractal_dimension:.3f}')
    print(f'  Hurst: {result.chaos.hurst_exponent:.3f}')
    
    print(f'\n?? GAME THEORY: {result.game_theory.strategy.value}')
    print(f'  Pareto Efficiency: {result.game_theory.pareto_efficiency:.2%}')
    print(f'  Dominant Prob: {result.game_theory.dominant_probability:.2%}')
    
    print(f'\n?? INFORMATION THEORY:')
    print(f'  Shannon Entropy: {result.information.shannon_entropy:.3f}')
    print(f'  SNR: {result.information.snr:.3f}')
    print(f'  KL Divergence: {result.information.kl_divergence:.3f}')
    
    print(f'\n?? RISK MATHEMATICS: {result.risk_math.risk_state.value}')
    print(f'  VaR 95%: ${result.risk_math.var_95:.2f}')
    print(f'  VaR 99%: ${result.risk_math.var_99:.2f}')
    print(f'  CVaR: ${result.risk_math.cvar:.2f}')
    print(f'  Max DD: {result.risk_math.max_drawdown_predicted:.2%}')
    
    if result.reasons:
        print(f'\n? REASONS:')
        for r in result.reasons:
            print(f'  {r}')
    
    if result.warnings:
        print(f'\n?? WARNINGS:')
        for w in result.warnings:
            print(f'  {w}')
    
    if result.insights:
        print(f'\n?? INSIGHTS:')
        for i in result.insights:
            print(f'  {i}')
    
    if result.prophecies:
        print(f'\n?? PROPHECIES:')
        for p in result.prophecies:
            print(f'  {p}')
    
    print('\n' + '='*80)
    
    # Test to_dict
    print('\n?? DICT OUTPUT:')
    print(result.to_dict())
    
except Exception as e:
    print(f'? ERROR: {e}')
    import traceback
    traceback.print_exc()
