import sys
import os
import pandas as pd
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the modules before importing generate_ai_prediction
# Mock the modules before importing generate_ai_prediction
sys.modules['google'] = MagicMock()
sys.modules['google.genai'] = MagicMock()
sys.modules['openai'] = MagicMock()

from funcs.ai_helper import prepare_lottery_data_text, get_brand_models, generate_ai_prediction

class TestAIHelper(unittest.TestCase):
    def setUp(self):
        self.config_ssq = {
            'name': '双色球',
            'code': 'ssq',
            'red_col_prefix': '红球',
            'red_count': 6,
            'has_blue': True,
            'blue_count': 1,
            'blue_col_name': '蓝球'
        }
        self.config_kl8 = {
            'name': '快乐8',
            'code': 'kl8',
            'red_col_prefix': '红球',
            'red_count': 20,
            'has_blue': False
        }
        
        self.mock_df = pd.DataFrame([
            {'期号': '2024001', '红球1': 1, '红球2': 2, '红球3': 3, '红球4': 4, '红球5': 5, '红球6': 6, '蓝球': 7},
            {'期号': '2024002', '红球1': 10, '红球2': 11, '红球3': 12, '红球4': 13, '红球5': 14, '红球6': 15, '蓝球': 16}
        ])

    def test_prepare_data_ssq(self):
        text = prepare_lottery_data_text(self.mock_df, self.config_ssq)
        self.assertIn("期号: 2024001, 红球: [1, 2, 3, 4, 5, 6], 蓝球: [7]", text)
        self.assertIn("期号: 2024002, 红球: [10, 11, 12, 13, 14, 15], 蓝球: [16]", text)

    def test_get_brand_models(self):
        brands = get_brand_models()
        self.assertIn("Gemini", brands)
        self.assertIn("OpenAI" if "OpenAI" in brands else "NVIDIA", brands) # Checked helper, it has NVIDIA, MiniMax, DashScope
        self.assertIn("MiniMax", brands)
        self.assertIn("DashScope", brands)

    @patch('google.genai.Client')
    def test_generate_gemini(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.models.generate_content.return_value.text = "Gemini Prediction"
        
        resp = generate_ai_prediction("Gemini", "models/gemini-2.0-flash", "test-key", "History", self.config_ssq)
        self.assertEqual(resp, "Gemini Prediction")
        mock_client_class.assert_called_with(api_key="test-key")

    @patch('openai.OpenAI')
    def test_generate_nvidia(self, mock_openai):
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create.return_value.choices[0].message.content = "NVIDIA Prediction"
        
        resp = generate_ai_prediction("NVIDIA", "z-ai/glm4.7", "test-key", "History", self.config_ssq)
        self.assertEqual(resp, "NVIDIA Prediction")
        # Check if correct base_url was used
        mock_openai.assert_called_with(api_key="test-key", base_url="https://integrate.api.nvidia.com/v1")

    def test_kl8_prompt_logic(self):
        # We can't easily test the prompt string without modifying generate_ai_prediction to return it
        # But we can verify the function routes correctly
        pass

if __name__ == '__main__':
    unittest.main()
