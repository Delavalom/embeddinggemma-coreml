import Foundation
import MachO
import Darwin

/// Current `phys_footprint` of this process (what Xcode's memory gauge and jetsam use).
func physFootprint() -> UInt64 {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
  let kr = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
  }
  precondition(kr == KERN_SUCCESS, "task_info failed")
  return info.phys_footprint
}

/// malloc keeps freed pages resident; ask it to hand them back so a baseline is honest.
func returnFreedMemoryToOS() { _ = malloc_zone_pressure_relief(nil, 0) }

/// Polls `physFootprint()` on a background thread to catch transient peaks.
final class PeakSampler: @unchecked Sendable {
  private let lock = NSLock()
  private var peak: UInt64 = 0
  private var running = false
  private let done = DispatchSemaphore(value: 0)

  func start() {
    lock.lock(); running = true; peak = physFootprint(); lock.unlock()
    Thread.detachNewThread { [self] in
      while true {
        let f = physFootprint()
        lock.lock()
        peak = max(peak, f)
        let keepGoing = running
        lock.unlock()
        if !keepGoing { break }
        usleep(200)
      }
      done.signal()
    }
  }

  func stop() -> UInt64 {
    lock.lock(); running = false; lock.unlock()
    done.wait()
    lock.lock(); defer { lock.unlock() }
    return max(peak, physFootprint())
  }
}
