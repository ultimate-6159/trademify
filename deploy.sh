#!/bin/bash
# Trademify - Full Deployment Script
# Deploy Frontend to Firebase Hosting + Backend to Cloud Run

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     Trademify Deployment Script            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-trademify}"
REGION="${REGION:-asia-southeast1}"
SERVICE_NAME="trademify-api"

# Check prerequisites
check_prerequisites() {
    echo -e "\n${YELLOW}📋 Checking prerequisites...${NC}"
    
    if ! command -v gcloud &> /dev/null; then
        echo -e "${RED}❌ gcloud CLI not found. Install: https://cloud.google.com/sdk/docs/install${NC}"
        exit 1
    fi
    
    if ! command -v firebase &> /dev/null; then
        echo -e "${RED}❌ Firebase CLI not found. Install: npm install -g firebase-tools${NC}"
        exit 1
    fi
    
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ Node.js not found${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
}

# Login to services
login_services() {
    echo -e "\n${YELLOW}🔐 Logging in to services...${NC}"
    
    # Check if already logged in
    if ! gcloud auth list 2>&1 | grep -q "ACTIVE"; then
        gcloud auth login
    fi
    
    if ! firebase login:list 2>&1 | grep -q "@"; then
        firebase login
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    firebase use $PROJECT_ID 2>/dev/null || firebase use --add
    
    echo -e "${GREEN}✅ Logged in successfully${NC}"
}

# Build Frontend
build_frontend() {
    echo -e "\n${YELLOW}🔨 Building Frontend...${NC}"
    
    cd frontend
    
    # Install dependencies
    npm install
    
    # Create production .env
    cat > .env.production << EOF
VITE_API_URL=https://${SERVICE_NAME}-${PROJECT_ID}.${REGION}.run.app
VITE_FIREBASE_API_KEY=\${FIREBASE_API_KEY}
VITE_FIREBASE_PROJECT_ID=${PROJECT_ID}
EOF
    
    # Build
    npm run build
    
    cd ..
    
    echo -e "${GREEN}✅ Frontend built successfully${NC}"
}

# Deploy Backend to Cloud Run
deploy_backend() {
    echo -e "\n${YELLOW}🚀 Deploying Backend to Cloud Run...${NC}"
    
    cd backend
    
    # Enable required APIs
    gcloud services enable run.googleapis.com
    gcloud services enable cloudbuild.googleapis.com
    
    # Build and deploy
    gcloud run deploy $SERVICE_NAME \
        --source . \
        --dockerfile Dockerfile.cloudrun \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 1Gi \
        --cpu 1 \
        --min-instances 1 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FIREBASE_PROJECT_ID=$PROJECT_ID"
    
    cd ..
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')
    echo -e "${GREEN}✅ Backend deployed: $SERVICE_URL${NC}"
}

# Deploy Frontend to Firebase Hosting
deploy_frontend() {
    echo -e "\n${YELLOW}🌐 Deploying Frontend to Firebase Hosting...${NC}"
    
    firebase deploy --only hosting
    
    echo -e "${GREEN}✅ Frontend deployed${NC}"
}

# Deploy Firestore rules
deploy_firestore() {
    echo -e "\n${YELLOW}📦 Deploying Firestore rules...${NC}"
    
    firebase deploy --only firestore:rules
    
    echo -e "${GREEN}✅ Firestore rules deployed${NC}"
}

# Main deployment
main() {
    check_prerequisites
    
    echo -e "\n${YELLOW}Select deployment option:${NC}"
    echo "1) Full deployment (Backend + Frontend)"
    echo "2) Frontend only (Firebase Hosting)"
    echo "3) Backend only (Cloud Run)"
    echo "4) Firestore rules only"
    read -p "Enter choice [1-4]: " choice
    
    case $choice in
        1)
            login_services
            deploy_backend
            build_frontend
            deploy_frontend
            deploy_firestore
            ;;
        2)
            login_services
            build_frontend
            deploy_frontend
            ;;
        3)
            login_services
            deploy_backend
            ;;
        4)
            login_services
            deploy_firestore
            ;;
        *)
            echo -e "${RED}Invalid choice${NC}"
            exit 1
            ;;
    esac
    
    echo -e "\n${GREEN}╔════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║     🎉 Deployment Complete!                ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
    
    # Print URLs
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)' 2>/dev/null || echo "N/A")
    HOSTING_URL="https://${PROJECT_ID}.web.app"
    
    echo -e "\n${BLUE}📊 Deployment URLs:${NC}"
    echo -e "   Frontend: ${HOSTING_URL}"
    echo -e "   Backend API: ${SERVICE_URL}"
    echo -e "   API Docs: ${SERVICE_URL}/docs"
}

# Run
main "$@"
