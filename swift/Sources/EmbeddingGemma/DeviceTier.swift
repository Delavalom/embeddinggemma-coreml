import Foundation

/// Which image budget a device gets by default (`ImageBudget.auto`): 280 soft tokens on A17 Pro and newer iPhones
/// and on Macs, 70 otherwise.
public enum DeviceTier {
  /// The first iPhone with an A17 Pro is iPhone16,1 (iPhone 15 Pro).
  static let firstFastIPhoneMajor = 16

  public static func defaultBudget() -> Int {
    #if os(macOS)
      return budget(machine: machine(), isMac: true)
    #else
      let p = ProcessInfo.processInfo
      return budget(machine: machine(), isMac: p.isiOSAppOnMac || p.isMacCatalystApp)
    #endif
  }

  /// The budget for a `hw.machine` string such as "iPhone16,1".
  static func budget(machine: String, isMac: Bool) -> Int {
    if isMac { return 280 }
    guard machine.hasPrefix("iPhone"),
          let major = Int(machine.dropFirst("iPhone".count).prefix { $0 != "," }), major >= firstFastIPhoneMajor
    else { return 70 }
    return 280
  }

  /// `hw.machine` (on the simulator, the simulated model from SIMULATOR_MODEL_IDENTIFIER).
  static func machine() -> String {
    if let sim = ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"] { return sim }
    var size = 0
    guard sysctlbyname("hw.machine", nil, &size, nil, 0) == 0, size > 0 else { return "" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname("hw.machine", &buf, &size, nil, 0) == 0 else { return "" }
    return String(decoding: buf.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
  }
}
