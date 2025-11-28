#!/bin/bash
# ============================================================================
# Create Sample Hindi Corpus for Testing
# ============================================================================
# This script creates an example corpus structure with sample Hindi sentences
# Use this to test the translation before using your real data

echo "======================================================================"
echo "CREATING SAMPLE HINDI CORPUS"
echo "======================================================================"
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p sample_hindi_corpus/news
mkdir -p sample_hindi_corpus/literature
mkdir -p sample_hindi_corpus/social
mkdir -p sample_hindi_corpus/education

echo "✅ Directories created"
echo ""

# Create news articles
echo "Creating sample news articles..."

cat > sample_hindi_corpus/news/politics.txt << 'EOF'
आज संसद में महत्वपूर्ण बहस हुई।
प्रधानमंत्री ने नई योजना की घोषणा की।
विपक्ष ने सरकार की आलोचना की।
कई सांसदों ने अपनी बात रखी।
बजट पर चर्चा जारी रहेगी।
EOF

cat > sample_hindi_corpus/news/sports.txt << 'EOF'
भारतीय क्रिकेट टीम ने मैच जीता।
खिलाड़ियों ने शानदार प्रदर्शन किया।
कप्तान बहुत खुश हैं।
अगला मैच अगले सप्ताह होगा।
दर्शक बहुत उत्साहित थे।
EOF

cat > sample_hindi_corpus/news/economy.txt << 'EOF'
देश की अर्थव्यवस्था बढ़ रही है।
नई नौकरियां उपलब्ध हैं।
निवेश में वृद्धि हुई है।
बाजार में तेजी देखी गई।
विशेषज्ञ सकारात्मक हैं।
EOF

# Create literature samples
echo "Creating sample literature..."

cat > sample_hindi_corpus/literature/poetry.txt << 'EOF'
चांद की रोशनी में सपने बुनते हैं।
तारे आसमान में चमकते हैं।
पक्षी सुबह गीत गाते हैं।
फूल बगीचे में खिलते हैं।
हवा धीरे-धीरे बहती है।
EOF

cat > sample_hindi_corpus/literature/story.txt << 'EOF'
एक गांव में एक किसान रहता था।
उसके पास एक छोटा खेत था।
वह बहुत मेहनती था।
हर दिन वह खेत में काम करता था।
उसका परिवार बहुत खुश था।
EOF

# Create social media samples
echo "Creating sample social media content..."

cat > sample_hindi_corpus/social/twitter.txt << 'EOF'
आज मौसम बहुत अच्छा है।
मैं बाजार जा रहा हूं।
क्या आप मेरी मदद कर सकते हैं?
यह किताब बहुत अच्छी है।
मुझे यह फिल्म पसंद आई।
EOF

cat > sample_hindi_corpus/social/facebook.txt << 'EOF'
दोस्तों के साथ समय बिताया।
आज एक नई जगह देखी।
खाना बहुत स्वादिष्ट था।
सभी को शुभकामनाएं।
अगले हफ्ते मिलेंगे।
EOF

# Create education samples
echo "Creating sample education content..."

cat > sample_hindi_corpus/education/lesson1.txt << 'EOF'
गणित एक महत्वपूर्ण विषय है।
छात्रों को नियमित अभ्यास करना चाहिए।
शिक्षक बहुत सहायक हैं।
कक्षा में सभी ध्यान से सुनते हैं।
परीक्षा अगले महीने होगी।
EOF

cat > sample_hindi_corpus/education/lesson2.txt << 'EOF'
विज्ञान हमारे जीवन में महत्वपूर्ण है।
प्रयोग करके हम सीखते हैं।
पुस्तकालय में कई किताबें हैं।
छात्रों को प्रश्न पूछने चाहिए।
शिक्षा भविष्य की कुंजी है।
EOF

echo "✅ Sample files created"
echo ""

# Show structure
echo "======================================================================"
echo "CORPUS STRUCTURE"
echo "======================================================================"
echo ""

if command -v tree &> /dev/null; then
    tree sample_hindi_corpus/
else
    find sample_hindi_corpus -type f | sort
fi

echo ""

# Count sentences
total_files=$(find sample_hindi_corpus -name "*.txt" | wc -l)
total_sentences=$(find sample_hindi_corpus -name "*.txt" -exec wc -l {} + | tail -1 | awk '{print $1}')

echo "======================================================================"
echo "STATISTICS"
echo "======================================================================"
echo "Domains: 4 (news, literature, social, education)"
echo "Files: $total_files"
echo "Total sentences: $total_sentences"
echo ""

# Calculate approximate cost
chars_per_sent=50
total_chars=$((total_sentences * chars_per_sent))
total_tokens=$((total_chars / 4))
cost_per_lang=$(echo "scale=2; ($total_tokens / 1000000 * 0.05) + ($total_tokens / 1000000 * 0.40)" | bc)
total_cost=$(echo "scale=2; $cost_per_lang * 2" | bc)
total_cost_inr=$(echo "scale=0; $total_cost * 85 / 1" | bc)

echo "Estimated cost for both languages:"
echo "  USD: \$$total_cost"
echo "  INR: ₹$total_cost_inr"
echo ""

echo "======================================================================"
echo "NEXT STEPS"
echo "======================================================================"
echo ""
echo "1. Test the translation:"
echo "   python translate_domain_corpus.py"
echo ""
echo "2. When prompted, enter:"
echo "   Source directory: sample_hindi_corpus"
echo "   Output directory: sample_translations"
echo ""
echo "3. Check results:"
echo "   ls -la sample_translations/bhojpuri/"
echo "   cat sample_translations/bhojpuri/news/politics.txt"
echo ""
echo "4. If successful, use with your real corpus!"
echo ""
echo "======================================================================"
echo "✅ Sample corpus ready!"
echo "======================================================================"
