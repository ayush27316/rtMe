//
//  WebSocketManager.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//

import Foundation
import SwiftUI
import Foundation
import Foundation

class WebSocketManager: ObservableObject {
    private var webSocket: URLSessionWebSocketTask?
    @Published var isConnected = false
    var onReceiveImage: ((UIImage, Int) -> Void)?
    private var isReconnecting = false
    private var reconnectAttempts = 0
    private let maxReconnectAttempts = 5
    
    func connect() {
        guard !isConnected, !isReconnecting else { return }
        
        guard let url = URL(string: "wss://0a02-132-205-229-214.ngrok-free.app/ws") else {
            print("Invalid WebSocket URL")
            return
        }
        
        let session = URLSession(configuration: .default)
        webSocket = session.webSocketTask(with: url)
        
        // Setup ping to keep connection alive
        schedulePing()
        
        webSocket?.resume()
        isConnected = true
        reconnectAttempts = 0
        receiveMessage()
        
        print("WebSocket connecting to: \(url)")
    }
    
    func disconnect() {
        isConnected = false
        webSocket?.cancel(with: .normalClosure, reason: nil)
        webSocket = nil
    }
    
    private func schedulePing() {
        guard let webSocket = webSocket else { return }
        
        webSocket.sendPing { [weak self] error in
            if let error = error {
                print("WebSocket ping failed: \(error)")
                self?.handleDisconnection()
            } else {
                // Schedule next ping after 10 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 10) { [weak self] in
                    self?.schedulePing()
                }
            }
        }
    }
    
    private func handleDisconnection() {
        guard !isReconnecting else { return }
        
        isConnected = false
        isReconnecting = true
        
        if reconnectAttempts < maxReconnectAttempts {
            reconnectAttempts += 1
            print("Attempting to reconnect... (Attempt \(reconnectAttempts))")
            
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(reconnectAttempts)) { [weak self] in
                self?.disconnect()
                self?.connect()
                self?.isReconnecting = false
            }
        } else {
            print("Max reconnection attempts reached")
            isReconnecting = false
        }
    }
    
    func sendImage(_ image: UIImage) {
        guard isConnected else {
            print("WebSocket not connected, attempting to reconnect...")
            handleDisconnection()
            return
        }
        
        guard let imageData = image.jpegData(compressionQuality: 0.5) else {
            print("Failed to compress image")
            return
        }
        
        let base64String = imageData.base64EncodedString()
        let messageDict = ["image": base64String]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: messageDict),
              let jsonString = String(data: jsonData, encoding: .utf8) else {
            print("Failed to create JSON message")
            return
        }
        
        let wsMessage = URLSessionWebSocketTask.Message.string(jsonString)
        
        webSocket?.send(wsMessage) { [weak self] error in
            if let error = error {
                print("Error sending image: \(error)")
                self?.handleDisconnection()
            }
        }
    }
    
    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            guard let self = self else { return }
            
            switch result {
            case .success(let message):
                self.handleReceivedMessage(message)
                // Continue listening for messages
                self.receiveMessage()
                
            case .failure(let error):
                print("Error receiving message: \(error)")
                self.handleDisconnection()
            }
        }
    }
    
    private func handleReceivedMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            processReceivedText(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                processReceivedText(text)
            }
        @unknown default:
            break
        }
    }
    
    private func processReceivedText(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let imageBase64 = json["processed_image"] as? String,
              let imageData = Data(base64Encoded: imageBase64),
              let image = UIImage(data: imageData) else {
            print("Failed to process received message")
            return
        }
        
        let faceCount = json["face_count"] as? Int ?? 0
        
        DispatchQueue.main.async {
            self.onReceiveImage?(image, faceCount)
        }
    }
}
