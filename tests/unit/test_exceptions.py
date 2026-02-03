
import unittest
from better_ai.utils.exceptions import BetterAIError, ConfigError, DataError, ModelError, TrainingError, EvaluationError

class TestExceptions(unittest.TestCase):
    """Unit tests for the custom exception classes."""

    def test_better_ai_error(self):
        """Test that BetterAIError is raised correctly."""
        with self.assertRaises(BetterAIError):
            raise BetterAIError("A Better AI error occurred")

    def test_config_error(self):
        """Test that ConfigError is raised correctly."""
        with self.assertRaises(ConfigError):
            raise ConfigError("A configuration error occurred")

    def test_data_error(self):
        """Test that DataError is raised correctly."""
        with self.assertRaises(DataError):
            raise DataError("A data error occurred")

    def test_model_error(self):
        """Test that ModelError is raised correctly."""
        with self.assertRaises(ModelError):
            raise ModelError("A model error occurred")

    def test_training_error(self):
        """Test that TrainingError is raised correctly."""
        with self.assertRaises(TrainingError):
            raise TrainingError("A training error occurred")

    def test_evaluation_error(self):
        """Test that EvaluationError is raised correctly."""
        with self.assertRaises(EvaluationError):
            raise EvaluationError("An evaluation error occurred")

    def test_inheritance(self):
        """Test that all custom exceptions inherit from BetterAIError."""
        self.assertTrue(issubclass(ConfigError, BetterAIError))
        self.assertTrue(issubclass(DataError, BetterAIError))
        self.assertTrue(issubclass(ModelError, BetterAIError))
        self.assertTrue(issubclass(TrainingError, BetterAIError))
        self.assertTrue(issubclass(EvaluationError, BetterAIError))

if __name__ == '__main__':
    unittest.main()
