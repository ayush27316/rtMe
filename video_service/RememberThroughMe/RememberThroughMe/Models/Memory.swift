//
//  Memory.swift
//  RememberThroughMe
//
//  Created by Swagat Bhowmik on 2025-02-01.
//

import UIKit

struct Memory {
let id: UUID
let image: UIImage
let title: String
let description: String
let dateCreated: Date
let context: String
var reminders: [Reminder]
}

struct Reminder {
let id: UUID
let title: String
let date: Date
let isComplete: Bool
}
