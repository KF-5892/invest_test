#!/usr/bin/env python3
"""
Script to run all alpha strategies and generate performance results.
"""

import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime, timedelta
import sys
import os

# Add alpha module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from alpha import (
    AlphaEngine,
    ShortTermReversal,
    MomentumAlpha,
    VolatilityAdjustedMomentum,
    CrossSectionalMomentum,
    ZScoreAlpha,
    RankAlpha,
    CorrelationAlpha,
    VolatilityAlpha,
    RSIAlpha,
    BollingerBandsAlpha,
    AdaptiveVolatilityAlpha,
    TrendStrengthAlpha,
    MeanReversionScore,
    VolatilityBreakoutAlpha,
    RelativeStrengthAlpha,
    MultiTimeframeRSI,
    VolatilityRegimeAlpha,
    PairwiseMomentumAlpha,
    VolatilityMeanReversionAlpha,
    GapReversal,
    SimpleMA,
    VolatilityRank,
    PriceVelocity,
    RollingBeta,
    TrendConsistency,
    ReturnSkew,
    DrawdownRecovery,
    VolumePrice,
    SeasonalTrend,
    ReturnDensity,
    MomentumDecay,
    VolBreakout,
    MeanReversionSpeed,
    TailRisk,
    ReturnStability,
    SimpleMomentum,
    NoiseRatio,
    VolatilitySurfaceAlpha,
    FactorMomentumAlpha,
    RegimeSwitchingAlpha,
    InformationRatioAlpha,
    LeadLagAlpha,
    ConditionalVolatilityAlpha,
    JumpDetectionAlpha,
    MicrostructureAlpha,
    TermStructureAlpha,
    CarryAlpha,
    OptionsFlowAlpha,
    CrossSectionalDispersionAlpha,
    EconomicSurpriseAlpha,
    SentimentReversalAlpha,
    MomentumCrashAlpha,
    VolatilityClusteringAlpha,
    AsymmetricRiskAlpha,
    LiquidityAlpha,
    CrossFrequencyAlpha,
    AdaptiveSignalAlpha,
    QuantileRankReversal,
    VolatilityAdjustedSkewness,
    MomentumVolatilityDecoupling,
    CrossAssetMomentumSpillover,
    AdaptiveMeanReversionSpeed,
    VolatilityMeanReversionWithSkew,
    HighFrequencyReversal,
    CorrelationBreakdownAlpha,
    MacroRegimeMomentum,
    TailRiskParityAlpha,
    FractionalDifferencing,
    VolatilityRiskPremium,
    DynamicBetaAlpha,
    MomentumQualityFilter,
    MultiHorizonVolatility,
    EventDrivenReversal,
    AdaptiveVolatilityTargeting,
    CrossSectionalMomentumDecay,
    HighMomentVolatilityTrap,
    MeanReversionWithMomentumFilter,
    AsymmetricVolatilityAlpha,
    MomentumPersistenceAlpha,
    CrossSectionalVolumeMomentum,
    RegimeChangeDetectionAlpha,
    OptionsInspiredSkewAlpha,
    LiquidityStressAlpha,
    EarningsSurpriseMomentum,
    MacroCycleAlpha,
    CrossCorrelationBreakdownAlpha,
    VolatilitySurfaceTermStructure,
    VolatilityRiskPremiumDecay,
    CrossAssetContagionAlpha,
    MomentumReversalTiming,
    OptionsGreeksSimulation,
    FractalMarketEfficiency,
    EconomicRegimeTransition,
    RiskAdjustedCarry,
    MarketMakerSignal,
    EarningsQualityMomentum,
    VolatilityClusteringBreakout,
    QuantumInfoRatioAlpha,
    AdvancedMicrostructureAlpha,
    CrossSectionalMomentumDecayQuantum,
    QuantumDispersionAlpha,
    QuantumVolSurfaceAlpha,
    MultihorizonMomentumQuantum,
    VolatilityQuantumTunneling,
    QuantumMeanReversionHarmonic,
    AdaptiveQuantumMomentum,
    QuantumEntanglementAlpha,
    MarketStructureArbitrage,
    VolatilityClusteringPredictor,
    CrossSectionalMeanReversionSpeed,
    AdaptiveRiskParityAlpha,
    MomentumQualityScore,
    VolatilitySpilloverAlpha,
    RegimeAwareReversal,
    LiquidityDrivenMomentum,
    MacroSentimentAlpha,
    VolatilityTermStructureAlpha,
    VolumeWeightedMomentum,
    CrossSectionalVolatilityRank,
    AdaptiveTimeDecay,
    MomentumBreakdownDetection,
    VolatilityAdjustedCarry,
    InterTemporalArbitrage,
    MicroTrendReversal,
    CorrelationMomentum,
    RiskAdjustedMeanReversion,
    VolatilitySignalFiltering,
    MultiScaleEntropyFilter,
    BayesianRegimeSwitching,
    VolatilitySmileArbitrage,
    NeuralNetworkMomentum,
    CopulaBasedPairsTrading,
    WaveletTransformAlpha,
    OptimalTransportDistance,
    HiddenMarkovModel,
    FractalDimensionTrading,
    MachineLearningEnsemble
)

def create_all_alphas():
    """Create all alpha strategies with different parameters."""
    alphas = []
    
    # Momentum-based alphas
    alphas.extend([
        ShortTermReversal(3),
        ShortTermReversal(5),
        ShortTermReversal(10),
        MomentumAlpha(30, 3),
        MomentumAlpha(60, 5),
        MomentumAlpha(90, 10),
        VolatilityAdjustedMomentum(30, 20),
        VolatilityAdjustedMomentum(60, 30),
        CrossSectionalMomentum(20),
        CrossSectionalMomentum(40),
    ])
    
    # Statistical alphas
    alphas.extend([
        ZScoreAlpha(60, 20),
        ZScoreAlpha(90, 30),
        RankAlpha(20),
        RankAlpha(40),
        CorrelationAlpha(60),
        CorrelationAlpha(90),
        VolatilityAlpha(30, 0.015),
        VolatilityAlpha(30, 0.025),
        RSIAlpha(14),
        RSIAlpha(21),
    ])
    
    # Enhanced alphas
    alphas.extend([
        BollingerBandsAlpha(),
        AdaptiveVolatilityAlpha(),
        TrendStrengthAlpha(),
        MeanReversionScore(),
        VolatilityBreakoutAlpha(),
        RelativeStrengthAlpha(),
        MultiTimeframeRSI(),
        VolatilityRegimeAlpha(),
        PairwiseMomentumAlpha(),
        VolatilityMeanReversionAlpha(),
    ])
    
    # Simple alphas - 20 new strategies
    alphas.extend([
        GapReversal(0.02),
        SimpleMA(10),
        VolatilityRank(20, 60),
        PriceVelocity(5),
        RollingBeta(30),
        TrendConsistency(10),
        ReturnSkew(30),
        DrawdownRecovery(60),
        VolumePrice(20),
        SeasonalTrend(252),
        ReturnDensity(30, 0.005),
        MomentumDecay(5, 20),
        VolBreakout(20, 1.5),
        MeanReversionSpeed(15),
        TailRisk(60, 0.05),
        ReturnStability(30),
        SimpleMomentum(7),
        NoiseRatio(20),
        SimpleMA(20),
        VolatilityRank(15, 45),
    ])
    
    # Advanced alphas - 20 sophisticated strategies
    alphas.extend([
        VolatilitySurfaceAlpha(10, 30),
        VolatilitySurfaceAlpha(5, 20),
        FactorMomentumAlpha(60, 0.95),
        FactorMomentumAlpha(40, 0.9),
        RegimeSwitchingAlpha(60, 0.02),
        RegimeSwitchingAlpha(45, 0.025),
        InformationRatioAlpha(30, 60),
        InformationRatioAlpha(20, 40),
        LeadLagAlpha(3, 20),
        LeadLagAlpha(5, 30),
        ConditionalVolatilityAlpha(0.1, 0.8),
        ConditionalVolatilityAlpha(0.15, 0.75),
        JumpDetectionAlpha(3.0, 20),
        JumpDetectionAlpha(2.5, 15),
        MicrostructureAlpha(10),
        MicrostructureAlpha(15),
        TermStructureAlpha(5, 20, 60),
        TermStructureAlpha(3, 15, 45),
        CarryAlpha(60, 10),
        CarryAlpha(40, 15),
        OptionsFlowAlpha(20),
        OptionsFlowAlpha(25),
        CrossSectionalDispersionAlpha(30),
        CrossSectionalDispersionAlpha(20),
        EconomicSurpriseAlpha(5, 15),
        EconomicSurpriseAlpha(3, 10),
        SentimentReversalAlpha(20, 1.5),
        SentimentReversalAlpha(25, 2.0),
        MomentumCrashAlpha(60, -0.1),
        MomentumCrashAlpha(45, -0.15),
        VolatilityClusteringAlpha(10, 2.0),
        VolatilityClusteringAlpha(15, 1.8),
        AsymmetricRiskAlpha(30),
        AsymmetricRiskAlpha(20),
        LiquidityAlpha(20),
        LiquidityAlpha(15),
        CrossFrequencyAlpha(5, 20, 60),
        CrossFrequencyAlpha(3, 15, 45),
        AdaptiveSignalAlpha(60),
        AdaptiveSignalAlpha(40),
    ])
    
    # New alphas - 21 innovative strategies
    alphas.extend([
        QuantileRankReversal(30, 0.8),
        QuantileRankReversal(20, 0.75),
        VolatilityAdjustedSkewness(20, 30),
        VolatilityAdjustedSkewness(15, 25),
        MomentumVolatilityDecoupling(15, 20, 60),
        MomentumVolatilityDecoupling(10, 15, 45),
        CrossAssetMomentumSpillover(10, 20),
        CrossAssetMomentumSpillover(8, 15),
        AdaptiveMeanReversionSpeed(15, 60),
        AdaptiveMeanReversionSpeed(12, 45),
        VolatilityMeanReversionWithSkew(20, 30),
        VolatilityMeanReversionWithSkew(15, 25),
        HighFrequencyReversal(5, 0.01),
        HighFrequencyReversal(3, 0.015),
        CorrelationBreakdownAlpha(30, 60),
        CorrelationBreakdownAlpha(25, 50),
        MacroRegimeMomentum(60, 20),
        MacroRegimeMomentum(45, 15),
        TailRiskParityAlpha(60, 0.05),
        TailRiskParityAlpha(45, 0.1),
        FractionalDifferencing(0.4, 30),
        FractionalDifferencing(0.3, 25),
        VolatilityRiskPremium(10, 30, 60),
        VolatilityRiskPremium(8, 25, 50),
        DynamicBetaAlpha(30, 60),
        DynamicBetaAlpha(25, 45),
        MomentumQualityFilter(30, 60),
        MomentumQualityFilter(20, 45),
        MultiHorizonVolatility([5, 10, 20, 40]),
        MultiHorizonVolatility([3, 8, 15, 30]),
        EventDrivenReversal(2.5, 20),
        EventDrivenReversal(2.0, 15),
        AdaptiveVolatilityTargeting(0.015, 60),
        AdaptiveVolatilityTargeting(0.02, 45),
        CrossSectionalMomentumDecay(20, 5),
        CrossSectionalMomentumDecay(15, 4),
        HighMomentVolatilityTrap(0.02, 0.03),
        HighMomentVolatilityTrap(0.015, 0.025),
        MeanReversionWithMomentumFilter(5, 20),
        MeanReversionWithMomentumFilter(4, 15),
    ])
    
    # Innovative alphas - 10 new strategies
    alphas.extend([
        AsymmetricVolatilityAlpha(30),
        AsymmetricVolatilityAlpha(20),
        MomentumPersistenceAlpha(5, 20),
        MomentumPersistenceAlpha(3, 15),
        CrossSectionalVolumeMomentum(20, 10),
        CrossSectionalVolumeMomentum(15, 8),
        RegimeChangeDetectionAlpha(60, 20),
        RegimeChangeDetectionAlpha(45, 15),
        OptionsInspiredSkewAlpha(30, 0.1, 0.9),
        OptionsInspiredSkewAlpha(20, 0.15, 0.85),
        LiquidityStressAlpha(10, 20),
        LiquidityStressAlpha(8, 15),
        EarningsSurpriseMomentum(5, 15),
        EarningsSurpriseMomentum(3, 10),
        MacroCycleAlpha(63, 20),
        MacroCycleAlpha(42, 15),
        CrossCorrelationBreakdownAlpha(15, 60),
        CrossCorrelationBreakdownAlpha(10, 45),
        VolatilitySurfaceTermStructure(10, 30, 60),
        VolatilitySurfaceTermStructure(5, 20, 40),
    ])
    
    # Next-generation alphas - 10 cutting-edge strategies
    alphas.extend([
        VolatilityRiskPremiumDecay(5, 20, 0.95),
        VolatilityRiskPremiumDecay(3, 15, 0.9),
        CrossAssetContagionAlpha(10, 2.0),
        CrossAssetContagionAlpha(8, 1.8),
        MomentumReversalTiming(20, 14, 20),
        MomentumReversalTiming(15, 10, 15),
        OptionsGreeksSimulation(30, 0.02),
        OptionsGreeksSimulation(21, 0.025),
        FractalMarketEfficiency(14, 0.5),
        FractalMarketEfficiency(10, 0.4),
        EconomicRegimeTransition(60, 1.5),
        EconomicRegimeTransition(45, 1.2),
        RiskAdjustedCarry(40, 20, 0.1),
        RiskAdjustedCarry(30, 15, 0.15),
        MarketMakerSignal(10, 5),
        MarketMakerSignal(8, 4),
        EarningsQualityMomentum(20, 60),
        EarningsQualityMomentum(15, 45),
        VolatilityClusteringBreakout(15, 1.8),
        VolatilityClusteringBreakout(12, 1.6),
    ])
    
    # Quantum alphas - 10 advanced quantum-inspired strategies
    alphas.extend([
        QuantumInfoRatioAlpha(15, 35, 0.8),
        QuantumInfoRatioAlpha(20, 40, 0.7),
        AdvancedMicrostructureAlpha(12, 0.7, 1.2),
        AdvancedMicrostructureAlpha(15, 0.6, 1.3),
        CrossSectionalMomentumDecayQuantum(18, 6, 0.85),
        CrossSectionalMomentumDecayQuantum(15, 4, 0.9),
        QuantumDispersionAlpha(25, 60, 1.3),
        QuantumDispersionAlpha(30, 50, 1.2),
        QuantumVolSurfaceAlpha(7, 21, 42, 0.9),
        QuantumVolSurfaceAlpha(5, 15, 30, 0.85),
        MultihorizonMomentumQuantum([5, 15, 30], 0.3),
        MultihorizonMomentumQuantum([3, 10, 20], 0.4),
        VolatilityQuantumTunneling(20, 1.5, 0.7),
        VolatilityQuantumTunneling(15, 1.8, 0.6),
        QuantumMeanReversionHarmonic(14, 0.8, 1.2),
        QuantumMeanReversionHarmonic(10, 0.9, 1.0),
        AdaptiveQuantumMomentum(10, 30, 0.5),
        AdaptiveQuantumMomentum(8, 25, 0.6),
        QuantumEntanglementAlpha(30, 0.6, 0.8),
        QuantumEntanglementAlpha(25, 0.7, 0.7),
    ])
    
    # Professional alphas - 10 new professional-grade strategies
    alphas.extend([
        MarketStructureArbitrage(20, 0.01),
        MarketStructureArbitrage(15, 0.015),
        VolatilityClusteringPredictor(15, 1.5),
        VolatilityClusteringPredictor(20, 1.8),
        CrossSectionalMeanReversionSpeed(30, 0.95),
        CrossSectionalMeanReversionSpeed(25, 0.9),
        AdaptiveRiskParityAlpha(30, 60),
        AdaptiveRiskParityAlpha(25, 45),
        MomentumQualityScore(60, 20),
        MomentumQualityScore(45, 15),
        VolatilitySpilloverAlpha(20, 3),
        VolatilitySpilloverAlpha(15, 2),
        RegimeAwareReversal(60, 5),
        RegimeAwareReversal(45, 4),
        LiquidityDrivenMomentum(30, 15),
        LiquidityDrivenMomentum(25, 12),
        MacroSentimentAlpha(45, 0.02),
        MacroSentimentAlpha(35, 0.025),
        VolatilityTermStructureAlpha(10, 30, 60),
        VolatilityTermStructureAlpha(8, 25, 50),
    ])
    
    # Custom alphas - 10 new innovative strategies
    alphas.extend([
        VolumeWeightedMomentum(30, 20),
        VolumeWeightedMomentum(20, 15),
        CrossSectionalVolatilityRank(20, 60),
        CrossSectionalVolatilityRank(15, 45),
        AdaptiveTimeDecay(30, 0.95),
        AdaptiveTimeDecay(20, 0.9),
        MomentumBreakdownDetection(10, 30, 0.6),
        MomentumBreakdownDetection(8, 25, 0.7),
        VolatilityAdjustedCarry(60, 30, 0.15),
        VolatilityAdjustedCarry(45, 25, 0.12),
        InterTemporalArbitrage(5, 25, 10),
        InterTemporalArbitrage(3, 20, 8),
        MicroTrendReversal(3, 15, 0.01),
        MicroTrendReversal(2, 12, 0.015),
        CorrelationMomentum(30, 20),
        CorrelationMomentum(25, 15),
        RiskAdjustedMeanReversion(20, 60, 2.0),
        RiskAdjustedMeanReversion(15, 45, 1.5),
        VolatilitySignalFiltering(20, 60, 0.02),
        VolatilitySignalFiltering(15, 45, 0.025),
    ])
    
    # Elite alphas - 10 new cutting-edge strategies
    alphas.extend([
        MultiScaleEntropyFilter([5, 10, 20], 0.6),
        MultiScaleEntropyFilter([3, 8, 15], 0.5),
        BayesianRegimeSwitching(30, 0.7),
        BayesianRegimeSwitching(45, 0.8),
        VolatilitySmileArbitrage(20, 10),
        VolatilitySmileArbitrage(15, 8),
        NeuralNetworkMomentum(10, 5, 0.01),
        NeuralNetworkMomentum(8, 4, 0.02),
        CopulaBasedPairsTrading(60, 0.5),
        CopulaBasedPairsTrading(45, 0.6),
        WaveletTransformAlpha(3, 'db4'),
        WaveletTransformAlpha(4, 'haar'),
        OptimalTransportDistance(30, 0.1),
        OptimalTransportDistance(20, 0.15),
        HiddenMarkovModel(3, 60),
        HiddenMarkovModel(2, 45),
        FractalDimensionTrading(50, 1.5),
        FractalDimensionTrading(40, 1.4),
        MachineLearningEnsemble(5, 20),
        MachineLearningEnsemble(7, 25),
    ])
    
    return alphas

def run_all_alphas():
    """Run all alpha strategies and save results."""
    
    # Initialize engine
    engine = AlphaEngine('data/48_Industry_Portfolios_Daily.csv')
    engine.load_data()
    
    # Get recent 3 years of data for evaluation
    end_date = engine.data.index[-1]
    start_date = end_date - timedelta(days=3*365)
    
    print(f"Running alphas from {start_date.date()} to {end_date.date()}")
    
    # Create all alphas
    alphas = create_all_alphas()
    
    # Register and run each alpha
    results = {}
    
    for i, alpha in enumerate(alphas):
        print(f"Running alpha {i+1}/{len(alphas)}: {alpha.name}")
        
        try:
            engine.register_alpha(alpha)
            engine.run_alpha(alpha.name, start_date=start_date.strftime('%Y-%m-%d'))
            
            # Calculate performance metrics
            metrics = engine.calculate_performance_metrics(alpha.name)
            
            results[alpha.name] = {
                'description': alpha.description,
                'metrics': metrics,
                'returns_data': engine.results[alpha.name]['returns'].to_list(),
                'dates': [d.strftime('%Y-%m-%d') for d in engine.results[alpha.name]['returns'].index]
            }
            
            print(f"  - Sharpe: {metrics['sharpe_ratio']:.3f}, Annual Return: {metrics['annualized_return']:.3f}")
            
        except Exception as e:
            print(f"  - Error: {e}")
            results[alpha.name] = {'error': str(e)}
    
    # Save results to JSON
    with open('alpha_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to alpha_results.json")
    
    # Save performance metrics to CSV
    save_performance_csv(results)
    
    # Create leaderboard
    create_leaderboard(results)
    
    return results

def save_performance_csv(results):
    """Save alpha performance metrics to CSV format."""
    
    # Prepare data for CSV
    csv_data = []
    
    for alpha_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            csv_data.append({
                'alpha_name': alpha_name,
                'description': result.get('description', ''),
                'total_return': metrics.get('total_return', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'volatility': metrics.get('volatility', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'num_trades': metrics.get('num_trades', 0)
            })
        else:
            # Include failed runs with error info
            csv_data.append({
                'alpha_name': alpha_name,
                'description': f"Error: {result.get('error', 'Unknown error')}",
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'num_trades': 0
            })
    
    # Sort by sharpe ratio
    csv_data.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    # Save to CSV
    csv_filename = 'alpha_performance.csv'
    fieldnames = ['alpha_name', 'description', 'sharpe_ratio', 'annualized_return', 
                  'total_return', 'volatility', 'max_drawdown', 'calmar_ratio', 'num_trades']
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Performance metrics saved to {csv_filename}")

def create_leaderboard(results):
    """Create a leaderboard of alpha performance."""
    
    leaderboard_data = []
    
    for alpha_name, result in results.items():
        if 'metrics' in result:
            metrics = result['metrics']
            leaderboard_data.append({
                'name': alpha_name,
                'description': result.get('description', ''),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'annualized_return': metrics.get('annualized_return', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'volatility': metrics.get('volatility', 0),
                'num_trades': metrics.get('num_trades', 0)
            })
    
    # Sort by Sharpe ratio
    leaderboard_data.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    # Save leaderboard
    with open('alpha_leaderboard.json', 'w') as f:
        json.dump(leaderboard_data, f, indent=2)
    
    # Print top performers
    print("\n" + "="*80)
    print("ALPHA LEADERBOARD - TOP 10 BY SHARPE RATIO")
    print("="*80)
    print(f"{'Rank':<4} {'Alpha Name':<35} {'Sharpe':<8} {'Ann.Ret':<8} {'Calmar':<8} {'MaxDD':<8}")
    print("-"*80)
    
    for i, alpha in enumerate(leaderboard_data[:10]):
        print(f"{i+1:<4} {alpha['name'][:34]:<35} "
              f"{alpha['sharpe_ratio']:<8.3f} "
              f"{alpha['annualized_return']:<8.3f} "
              f"{alpha['calmar_ratio']:<8.3f} "
              f"{alpha['max_drawdown']:<8.3f}")
    
    print("="*80)

if __name__ == "__main__":
    results = run_all_alphas()