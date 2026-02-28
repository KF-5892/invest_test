# Alpha Miners Dashboard

A modern web dashboard for visualizing and analyzing algorithmic trading strategies performance.

## Features

- 📊 Interactive performance charts using Recharts
- 🔍 Strategy leaderboard with detailed metrics
- 💻 Source code viewer with syntax highlighting
- 📱 Responsive design optimized for all devices
- ⚡ Static generation for fast loading on Vercel

## Tech Stack

- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Code Highlighting**: react-syntax-highlighter
- **Deployment**: Vercel
- **TypeScript**: Full type safety

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

1. Install dependencies:
```bash
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

### Building for Production

```bash
npm run build
```

This will generate a static export in the `out` directory, optimized for Vercel deployment.

## Project Structure

```
vercel-frontend/
├── app/                    # Next.js app directory
│   ├── components/         # React components
│   │   ├── ui/            # Base UI components
│   │   ├── AlphaCard.tsx  # Strategy card component
│   │   └── CodeViewer.tsx # Source code viewer
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Main dashboard page
├── public/
│   └── data/              # Static data files
│       ├── alpha_leaderboard.json
│       ├── source_code.json
│       ├── summary.json
│       └── metadata.json
├── next.config.js         # Next.js configuration
├── tailwind.config.js     # Tailwind configuration
└── vercel.json           # Vercel deployment configuration
```

## Data Files

The dashboard uses several JSON data files:

- `alpha_leaderboard.json`: Strategy performance metrics
- `source_code.json`: Source code for each strategy  
- `summary.json`: Aggregated statistics
- `metadata.json`: Dataset metadata

## Deployment

This project is optimized for deployment on Vercel:

1. Connect your GitHub repository to Vercel
2. Vercel will automatically detect Next.js and configure the build
3. The static export will be deployed to Vercel's global CDN

### Manual Deployment

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy to production
vercel --prod
```

## Performance Metrics

The dashboard displays the following key metrics for each strategy:

- **Sharpe Ratio**: Risk-adjusted returns
- **Annualized Return**: Yearly return percentage
- **Calmar Ratio**: Return vs maximum drawdown
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Number of Trades**: Total trading frequency

## License

MIT License - see LICENSE file for details.