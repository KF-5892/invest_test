#!/bin/bash

set -e

echo "Starting alpha processing and deployment..."

# Run alphas
echo "Running alphas..."
python run_alphas.py

python generate_source_code.py
# Generate Vercel data
echo "Generating Vercel data..."
python generate_vercel_data.py

# Move to frontend directory
cd vercel-frontend

# Install dependencies
echo "Installing npm dependencies..."
npm install

# Build the project
echo "Building the project..."
npm run build

# Deploy to Vercel
echo "Deploying to Vercel..."
npx vercel --prod

echo "Deployment completed successfully!"
