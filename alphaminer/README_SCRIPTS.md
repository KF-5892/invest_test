# Alpha Miners - Script Documentation

This document describes the key scripts for managing alpha strategies and generating frontend data.

## Main Scripts

### 1. `generate_source_code.py`
**Purpose**: Automatically extracts source code from all alpha strategy classes and generates `source_code.json` for the frontend.

**Usage**:
```bash
python generate_source_code.py
```

**What it does**:
- Scans all Python files in the `alpha/` directory
- Extracts source code for each alpha strategy class using AST parsing
- Generates `vercel-frontend/public/data/source_code.json` with:
  - Class name
  - Source file
  - Complete source code
  - Language identifier (Python)

**Output**: 
- `vercel-frontend/public/data/source_code.json` (~190KB)
- Console summary showing extracted classes by file

### 2. `generate_vercel_data.py`
**Purpose**: Comprehensive data generation for the Vercel frontend deployment.

**Usage**:
```bash
python generate_vercel_data.py
```

**What it does**:
1. **Calls `generate_source_code.py`** to create source code data
2. **Copies alpha leaderboard** from `alpha_leaderboard.json`
3. **Generates metadata** with strategy counts and timestamps
4. **Creates summary statistics** including top performers
5. **Generates cumulative returns** using real performance data
6. **Creates API-style data** for frontend consumption

**Generated Files**:
- `source_code.json` - All alpha source code (185.6 KB)
- `alpha_leaderboard.json` - Performance rankings (61.9 KB)
- `summary.json` - Summary statistics (1.7 KB)
- `cumulative_returns.json` - Time series data (199.3 KB)
- `metadata.json` - Metadata info (0.3 KB)
- `api.json` - Combined API data (252.6 KB)

### 3. `run_alphas.py`
**Purpose**: Execute all alpha strategies and generate performance results.

**Usage**:
```bash
python run_alphas.py
```

**What it does**:
- Loads 48 Industry Portfolio data
- Runs all registered alpha strategies (119 classes, 300+ instances)
- Calculates performance metrics (Sharpe, returns, drawdown, etc.)
- Generates output files for analysis

**Generated Files**:
- `alpha_results.json` - Detailed results for each alpha
- `alpha_performance.csv` - Performance metrics in CSV format
- `alpha_leaderboard.json` - Ranked performance data

## Workflow

### Complete Alpha Development Workflow:

1. **Develop new alphas** in `alpha/` directory
2. **Register alphas** in `alpha/__init__.py` and `run_alphas.py`
3. **Run backtests**: `python run_alphas.py`
4. **Generate frontend data**: `python generate_vercel_data.py`
5. **Deploy to Vercel**: `cd vercel-frontend && npm run build && vercel --prod`

### Quick Frontend Update:

If you only need to update source code (e.g., after adding new alphas):
```bash
python generate_source_code.py
```

If you need complete data refresh:
```bash
python generate_vercel_data.py
```

## Alpha Strategy Structure

### File Organization:
```
alpha/
├── __init__.py              # Import all alphas
├── base_alpha.py           # BaseAlpha class and AlphaEngine
├── momentum_alphas.py      # Momentum-based strategies
├── statistical_alphas.py   # Statistical approaches
├── enhanced_alphas.py      # Enhanced techniques
├── simple_alphas.py        # Simple trading strategies
├── advanced_alphas.py      # Sophisticated methods
├── new_alphas.py          # Recent innovations
├── innovative_alphas.py    # Creative approaches
├── next_gen_alphas.py     # Cutting-edge strategies
├── quantum_alphas.py      # Quantum-inspired methods
└── professional_alphas.py  # Professional-grade strategies
```

### Alpha Categories:
- **119 total classes** across 11 files
- **300+ strategy instances** with different parameters
- Categories: Momentum, Mean Reversion, Volatility, Cross-sectional, Regime-based, etc.

### Recent Additions (Professional Alphas):
1. `MarketStructureArbitrage` - Exploits market microstructure inefficiencies
2. `VolatilityClusteringPredictor` - Predicts volatility clustering patterns
3. `CrossSectionalMeanReversionSpeed` - Speed of mean reversion analysis
4. `AdaptiveRiskParityAlpha` - Dynamic risk parity with correlation adjustment
5. `MomentumQualityScore` - Quality-filtered momentum strategies
6. `VolatilitySpilloverAlpha` - Cross-asset volatility spillover effects
7. `RegimeAwareReversal` - Regime-dependent reversal strategies
8. `LiquidityDrivenMomentum` - Liquidity-filtered momentum
9. `MacroSentimentAlpha` - Economic sentiment-driven signals
10. `VolatilityTermStructureAlpha` - Volatility term structure exploitation

## Data Flow

```
Alpha Development → run_alphas.py → generate_vercel_data.py → Vercel Frontend
                                  → generate_source_code.py ↗
```

## Frontend Integration

The generated JSON files are consumed by the Next.js frontend in `vercel-frontend/`:

- **Source Code Viewer**: Uses `source_code.json`
- **Performance Charts**: Uses `cumulative_returns.json`
- **Leaderboard**: Uses `alpha_leaderboard.json`
- **Summary Stats**: Uses `summary.json`
- **API Endpoints**: Uses `api.json`

## Error Handling

- Scripts include comprehensive error handling
- Missing files generate warnings but don't stop execution
- Source code extraction gracefully handles parsing errors
- Fallback to simulated data when real data unavailable

## Performance Notes

- Source code extraction: ~1-2 seconds for 119 classes
- Full data generation: ~5-10 seconds depending on data size
- Generated files total: ~900KB compressed data
- Ready for production deployment

## Maintenance

To add new alpha strategies:
1. Create new class in appropriate `alpha/*.py` file
2. Add import to `alpha/__init__.py`
3. Register in `run_alphas.py` create_all_alphas() function
4. Update this README if adding new categories

The scripts automatically detect and include new strategies without modification.