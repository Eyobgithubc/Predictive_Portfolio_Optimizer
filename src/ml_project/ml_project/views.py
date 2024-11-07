# mlapi/views.py
import pickle
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PredictionView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load the trained model
        with open('notebook/model_24-09-2024-07-29-59-293895.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def post(self, request):
        data = request.data
        features = np.array(data['features']).reshape(1, -1)  # Reshape for model input
        prediction = self.model.predict(features)
        return Response({'prediction': prediction.tolist()}, status=status.HTTP_200_OK)
