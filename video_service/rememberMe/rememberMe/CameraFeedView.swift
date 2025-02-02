//
//  CameraFeedView.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//



import SwiftUI
import AVFoundation

struct CameraFeedView: View {
    @StateObject private var viewModel = CameraViewModel()
    
    var body: some View {
        ZStack {
            // Camera preview with processed image overlay
            CameraPreviewLayer(viewModel: viewModel)
                .ignoresSafeArea()
            
            // Status indicators
            VStack {
                Spacer()
                
                // Face count indicator
                if viewModel.faceCount > 0 {
                    Text("Faces detected: \(viewModel.faceCount)")
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.black.opacity(0.5))
                        .cornerRadius(10)
                }
                
                // Connection status
                HStack {
                    Circle()
                        .fill(viewModel.isConnectedToServer ? Color.green : Color.red)
                        .frame(width: 10, height: 10)
                    
                    Text(viewModel.isConnectedToServer ? "Connected" : "Disconnected")
                        .font(.caption)
                        .foregroundColor(.white)
                }
                .padding()
                .background(Color.black.opacity(0.5))
                .cornerRadius(20)
            }
            .padding()
            
            // Error message
            if let error = viewModel.error {
                Text(error)
                    .foregroundColor(.red)
                    .padding()
                    .background(Color.black.opacity(0.5))
                    .cornerRadius(10)
            }
        }
        .onAppear {
            viewModel.startSession()
        }
        .onDisappear {
            viewModel.stopSession()
        }
    }
}
