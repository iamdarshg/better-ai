
import unittest
from better_ai.utils.react_notebook import ReActNotebook

class TestReActNotebook(unittest.TestCase):
    def test_notebook(self):
        notebook = ReActNotebook()
        notebook.add_thought("I need to do something.")
        notebook.add_action("do_something", {"arg1": "value1"})
        notebook.add_observation("Something happened.")
        notebook.add_code("print('hello')")
        notebook.add_error("Something went wrong.")
        notebook.add_self_correction("I should have done something else.")

        json_output = notebook.to_json()

        new_notebook = ReActNotebook()
        new_notebook.from_json(json_output)

        self.assertEqual(len(notebook.trajectory), len(new_notebook.trajectory))
        for original, new in zip(notebook.trajectory, new_notebook.trajectory):
            self.assertEqual(original, new)

if __name__ == '__main__':
    unittest.main()
