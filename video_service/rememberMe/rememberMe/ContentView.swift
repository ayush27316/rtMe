//
//  ContentView.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//



import SwiftUI
import AVFoundation

struct ContentView: View {
    @State private var isAnalyzing = false
    @State private var showInput = false
    @State private var inputText = ""
    let backgroundColor = Color(red: 245/255, green: 245/255, blue: 245/255)
    
    var body: some View {
        VStack {
            ZStack {
                // Camera Feed View
                CameraFeedView()
                    .edgesIgnoringSafeArea(.all) // Make sure the camera feed ignores safe areas and takes up all screen space
                
                // Input Popup
                if showInput {
                    InputPopup(text: $inputText)
                        .padding()
                        .background(Color.white.cornerRadius(16))
                        .transition(.move(edge: .bottom))
                }
            }
            
            VStack {
                Spacer()
                Text("Recognizing...")
                    .font(.footnote)
                    .foregroundColor(.gray)
                
                ProgressView()
                    .progressViewStyle(CircularProgressViewStyle())
                    .scaleEffect(isAnalyzing ? 1 : 0)
                    .animation(.easeInOut, value: isAnalyzing)
                
                Spacer()
                
                // Buttons
                HStack(spacing: 16) {
                    Button(action: {
                        isAnalyzing = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                            isAnalyzing = false
                            showInput = true
                        }
                    }) {
                        Image(systemName: "camera.fill")
                            .frame(width: 45, height: 45)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .clipShape(Circle())
                    }
                    
                    Button(action: { print("Memory Book") }) {
                        Image(systemName: "book.fill")
                            .frame(width: 45, height: 45)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .clipShape(Circle())
                    }
                    
                    Button(action: { print("Voice Assistant") }) {
                        Image(systemName: "waveform.path.ecg")
                            .frame(width: 45, height: 45)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .clipShape(Circle())
                    }
                }
                .frame(maxWidth: .infinity)
                .frame(height: 60)
                .background(backgroundColor)
                .cornerRadius(16)
                .padding(.bottom)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(backgroundColor)
        .edgesIgnoringSafeArea(.all)
    }
}
