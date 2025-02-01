//
//  InputPopup.swift
//  rememberMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//

import SwiftUI

struct InputPopup: View {
    @Binding var text: String
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack {
            Text("Add Details")
                .font(.headline)
                .padding(.bottom, 8)
            
            TextField("Describe what you see...", text: $text)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .frame(maxWidth: 300)
                .padding([.top, .bottom])
            
            HStack(spacing: 16) {
                Button {
                    // Cancel and close the popup
                    self.presentationMode.wrappedValue.dismiss()
                    text = ""
                } label: {
                    Text("Cancel")
                        .frame(minWidth: 100)
                }
                
                Button {
                    // Save text to backend and close
                    print("Saved: \(text)")
                    text = ""
                    self.presentationMode.wrappedValue.dismiss()
                } label: {
                    Text("Save")
                        .frame(minWidth: 100)
                }
            }
        }
        .frame(maxWidth: 300, maxHeight: 200)
        .background(Color.white)
        .cornerRadius(16)
        .shadow(radius: 5)
        .padding()
        .background(
            Rectangle()
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color.black.opacity(0.2))
                .edgesIgnoringSafeArea(.all)
                .onTapGesture {
                    self.presentationMode.wrappedValue.dismiss()
                    text = ""
                }
        )
    }
}
