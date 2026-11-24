// Apple Vision OCR, batch mode. Reads image paths from stdin (one per line) and prints one JSON
// line per image: {"path": ..., "lines": [{"text", "conf", "x", "y", "w", "h"}], "error": ...}.
// Coordinates are normalised to [0,1] with the origin at the TOP-left.
//
//   swiftc -O tools/ocr/vision_ocr.swift -o tools/ocr/vision_ocr
//   ls *.jpg | tools/ocr/vision_ocr > ocr.jsonl

import Foundation
import Vision

struct Line: Codable { let text: String; let conf: Float; let x, y, w, h: Double }
struct Result: Codable { let path: String; let lines: [Line]; let error: String? }

func recognize(_ path: String) -> Result {
    let url = URL(fileURLWithPath: path)
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false  // invoices are IDs and numbers; "correction" corrupts them
    let handler = VNImageRequestHandler(url: url, options: [:])
    do {
        try handler.perform([request])
    } catch {
        return Result(path: path, lines: [], error: String(describing: error))
    }
    let lines: [Line] = (request.results ?? []).compactMap { obs in
        guard let top = obs.topCandidates(1).first else { return nil }
        let b = obs.boundingBox
        return Line(text: top.string, conf: top.confidence,
                    x: Double(b.minX), y: Double(1 - b.maxY), w: Double(b.width), h: Double(b.height))
    }
    return Result(path: path, lines: lines, error: nil)
}

let encoder = JSONEncoder()
while let raw = readLine() {
    let path = raw.trimmingCharacters(in: .whitespaces)
    if path.isEmpty { continue }
    let data = try! encoder.encode(recognize(path))
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}
