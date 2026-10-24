import Testing
@testable import EmbeddingGemma

@Suite struct DeviceTierTests {
  @Test(arguments: [
    ("iPhone16,1", 280),  // iPhone 15 Pro, A17 Pro
    ("iPhone16,2", 280),
    ("iPhone17,3", 280),
    ("iPhone18,1", 280),
    ("iPhone15,4", 70),  // iPhone 15, A16
    ("iPhone15,2", 70),  // iPhone 14 Pro
    ("iPhone13,2", 70),  // iPhone 12
    ("iPhone9,1", 70),
    ("iPad16,3", 70),  // only iPhones are mapped (spec section 8)
    ("arm64", 70),  // unknown
    ("", 70),
    ("iPhone", 70),
    ("iPhoneX,1", 70),
  ])
  func iOSBudgetFromMachine(machine: String, budget: Int) {
    #expect(DeviceTier.budget(machine: machine, isMac: false) == budget)
  }

  @Test func macIsAlways280() {
    #expect(DeviceTier.budget(machine: "arm64", isMac: true) == 280)
    #expect(DeviceTier.budget(machine: "iPhone13,2", isMac: true) == 280)
  }

  @Test func thisMacGets280() {
    #expect(DeviceTier.defaultBudget() == 280)
    #expect(!DeviceTier.machine().isEmpty)
  }

  @Test func autoBudgetResolves() {
    #expect(EmbeddingGemma.ImageBudget.tokens70.resolved == 70)
    #expect(EmbeddingGemma.ImageBudget.tokens280.resolved == 280)
    #expect(EmbeddingGemma.ImageBudget.auto.resolved == DeviceTier.defaultBudget())
  }
}
