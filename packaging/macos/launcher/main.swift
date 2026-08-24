// Vitruve.app's launcher.
//
// It owns exactly one thing: the lifetime of the Python server. A user who
// double-clicks the app gets a window, a progress bar for the first-run weight
// download, a browser tab, and a Quit that actually stops the server. Nothing
// in here knows what a landmark is.
//
// Why a compiled launcher rather than a shell script in Contents/MacOS:
//
//   * a script cannot own a Dock icon, a menu bar item, or a window, so there
//     would be no visible way to quit and no way to show progress;
//   * a script's child would outlive a force-quit of the wrapper;
//   * every Mach-O in a notarised bundle has to be signed anyway, and the
//     shell would be /bin/sh outside the bundle, which means the app has no
//     main executable of its own to attach entitlements to.
//
// The port is chosen at runtime. 8731 is the documented default for
// `vitruve serve` and is exactly the wrong thing to hard-code here: a user who
// already has one running from a terminal would get a bind failure on launch
// with nowhere to read the error.

import AppKit
import Foundation

// MARK: - Report palette

enum Palette {
    static let ivory = NSColor(calibratedRed: 0.953, green: 0.941, blue: 0.910, alpha: 1)
    static let ink = NSColor(calibratedRed: 0.102, green: 0.098, blue: 0.086, alpha: 1)
    static let rule = NSColor(calibratedRed: 0.588, green: 0.569, blue: 0.522, alpha: 1)
}

// MARK: - Free port

/// Ask the kernel for a port instead of guessing one.
///
/// Binding port 0 and reading the assignment back is the only way to get a
/// port that is free right now. There is a race between closing this socket
/// and uvicorn binding it, which is unavoidable without passing the listening
/// fd through, and is narrow enough that the practical failure mode is another
/// program deliberately hunting for the same port.
func freeLoopbackPort() -> UInt16? {
    let fd = socket(AF_INET, SOCK_STREAM, 0)
    if fd < 0 { return nil }
    defer { close(fd) }

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = 0
    addr.sin_addr.s_addr = INADDR_LOOPBACK.bigEndian
    addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)

    let bound = withUnsafePointer(to: &addr) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    if bound != 0 { return nil }

    var out = sockaddr_in()
    var len = socklen_t(MemoryLayout<sockaddr_in>.size)
    let got = withUnsafeMutablePointer(to: &out) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            getsockname(fd, $0, &len)
        }
    }
    if got != 0 { return nil }
    return UInt16(bigEndian: out.sin_port)
}

// MARK: - Logging

/// A log outside the bundle, because the bundle is signed and sealed and
/// anything written inside it invalidates the signature.
final class Log {
    static let shared = Log()
    private let handle: FileHandle?
    private let queue = DispatchQueue(label: "us.ericspencer.vitruve.log")

    init() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Vitruve", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("vitruve.log")
        if !FileManager.default.fileExists(atPath: file.path) {
            FileManager.default.createFile(atPath: file.path, contents: nil)
        }
        handle = try? FileHandle(forWritingTo: file)
        _ = try? handle?.seekToEnd()
    }

    func write(_ line: String) {
        let stamped = ISO8601DateFormatter().string(from: Date()) + "  " + line + "\n"
        FileHandle.standardError.write(Data(stamped.utf8))
        queue.async { [handle] in
            try? handle?.write(contentsOf: Data(stamped.utf8))
        }
    }
}

// MARK: - The server process

/// Spawns and supervises `Contents/Resources/app_main.py`.
final class Server {
    private let process = Process()
    private let stdoutPipe = Pipe()
    private let stderrPipe = Pipe()
    // Held open for the child's whole life. Closing it is the signal the
    // child's watchdog thread uses to exit, which is what stops a force-quit
    // of this launcher from orphaning a server.
    private let stdinPipe = Pipe()

    private var stdoutRemainder = Data()
    let port: UInt16
    private(set) var recentStderr: [String] = []

    var onEvent: (([String: Any]) -> Void)?
    var onExit: ((Int32) -> Void)?

    init?(resources: URL, port: UInt16) {
        self.port = port
        let python = resources.appendingPathComponent("runtime/bin/python3.11")
        let entry = resources.appendingPathComponent("app_main.py")
        guard FileManager.default.isExecutableFile(atPath: python.path),
              FileManager.default.fileExists(atPath: entry.path) else { return nil }

        process.executableURL = python
        process.arguments = [
            // -I isolates the interpreter from a user's PYTHONPATH and from a
            // site-packages in their home directory. A bundled app that picks
            // up a stray numpy from ~/.local is a support ticket nobody can
            // reproduce. -u keeps the event stream unbuffered.
            "-I", "-u",
            entry.path,
            "--port", String(port),
            "--resources", resources.path,
        ]
        var env = ProcessInfo.processInfo.environment
        env["VITRUVE_BUNDLED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = env
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
    }

    func start() throws {
        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            self?.consumeStdout(handle.availableData)
        }
        stderrPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
                Log.shared.write("server: " + line)
                DispatchQueue.main.async {
                    self?.recentStderr.append(String(line))
                    if let n = self?.recentStderr.count, n > 40 {
                        self?.recentStderr.removeFirst(n - 40)
                    }
                }
            }
        }
        process.terminationHandler = { [weak self] proc in
            DispatchQueue.main.async { self?.onExit?(proc.terminationStatus) }
        }
        try process.run()
        Log.shared.write("server: pid \(process.processIdentifier) on port \(port)")
    }

    private func consumeStdout(_ data: Data) {
        guard !data.isEmpty else { return }
        stdoutRemainder.append(data)
        let newline = UInt8(ascii: "\n")
        while let idx = stdoutRemainder.firstIndex(of: newline) {
            let lineData = stdoutRemainder[stdoutRemainder.startIndex..<idx]
            stdoutRemainder.removeSubrange(stdoutRemainder.startIndex...idx)
            guard let line = String(data: Data(lineData), encoding: .utf8) else { continue }
            // uvicorn's access log shares this stream, so anything without the
            // sentinel is a log line and not an event.
            let sentinel = "@@VITRUVE@@ "
            guard line.hasPrefix(sentinel) else {
                if !line.isEmpty { Log.shared.write("server: " + line) }
                continue
            }
            let json = String(line.dropFirst(sentinel.count))
            guard let obj = try? JSONSerialization.jsonObject(with: Data(json.utf8)),
                  let dict = obj as? [String: Any] else { continue }
            Log.shared.write("event: " + json)
            DispatchQueue.main.async { self.onEvent?(dict) }
        }
    }

    var isRunning: Bool { process.isRunning }

    /// SIGTERM, then SIGKILL if it does not go.
    ///
    /// Blocking the main thread here is deliberate: this runs from
    /// `applicationWillTerminate`, and returning before the child is gone is
    /// how an orphan happens.
    func stop(graceSeconds: Double = 6.0) {
        guard process.isRunning else { return }
        try? stdinPipe.fileHandleForWriting.close()
        process.terminate()

        let deadline = Date().addingTimeInterval(graceSeconds)
        while process.isRunning && Date() < deadline {
            usleep(50_000)
        }
        if process.isRunning {
            Log.shared.write("server did not stop on SIGTERM; sending SIGKILL")
            kill(process.processIdentifier, SIGKILL)
            process.waitUntilExit()
        }
        Log.shared.write("server stopped")
    }
}

// MARK: - Health polling

/// Poll `/health` until it answers, then hand back the URL.
///
/// "Open the browser when the process starts" would open a tab on a connection
/// refused, because importing torch takes seconds. The readiness signal is the
/// endpoint answering, not the process existing.
func waitForHealth(port: UInt16, timeout: TimeInterval, onReady: @escaping (Bool) -> Void) {
    let url = URL(string: "http://127.0.0.1:\(port)/health")!
    let deadline = Date().addingTimeInterval(timeout)
    let session = URLSession(configuration: .ephemeral)

    func attempt() {
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        session.dataTask(with: req) { _, response, _ in
            if let http = response as? HTTPURLResponse, http.statusCode == 200 {
                DispatchQueue.main.async { onReady(true) }
                return
            }
            if Date() >= deadline {
                DispatchQueue.main.async { onReady(false) }
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.4) { attempt() }
        }.resume()
    }
    attempt()
}

// MARK: - Window

final class StatusWindow: NSWindow {
    let headline = NSTextField(labelWithString: "Starting Vitruve")
    let detail = NSTextField(wrappingLabelWithString: "")
    let bar = NSProgressIndicator()
    let openButton = NSButton(title: "Open in browser", target: nil, action: nil)
    let quitButton = NSButton(title: "Quit", target: nil, action: nil)

    init() {
        super.init(
            contentRect: NSRect(x: 0, y: 0, width: 460, height: 210),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        title = "Vitruve"
        isReleasedWhenClosed = false
        center()

        let content = NSView(frame: contentRect(forFrameRect: frame))
        content.wantsLayer = true
        content.layer?.backgroundColor = Palette.ivory.cgColor
        contentView = content

        headline.font = NSFont.systemFont(ofSize: 15, weight: .medium)
        headline.textColor = Palette.ink
        detail.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        detail.textColor = Palette.rule
        detail.maximumNumberOfLines = 3

        bar.isIndeterminate = true
        bar.style = .bar
        bar.controlSize = .small
        bar.startAnimation(nil)

        openButton.bezelStyle = .rounded
        openButton.isEnabled = false
        quitButton.bezelStyle = .rounded

        let buttons = NSStackView(views: [openButton, quitButton])
        buttons.orientation = .horizontal
        buttons.spacing = 10

        let stack = NSStackView(views: [headline, bar, detail, buttons])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 14
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 28),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -28),
            stack.centerYAnchor.constraint(equalTo: content.centerYAnchor),
            bar.widthAnchor.constraint(equalTo: stack.widthAnchor),
        ])
    }
}

// MARK: - Delegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var server: Server?
    private var statusItem: NSStatusItem?
    private let window = StatusWindow()
    private var url: URL?
    private var quitting = false

    func applicationDidFinishLaunching(_ note: Notification) {
        NSApp.setActivationPolicy(.regular)
        buildMenu()
        buildStatusItem()

        window.openButton.target = self
        window.openButton.action = #selector(openBrowser)
        window.quitButton.target = self
        window.quitButton.action = #selector(quit)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        guard let resources = Bundle.main.resourceURL else {
            fail("This build has no Resources directory. The bundle is incomplete.")
            return
        }
        guard let port = freeLoopbackPort() else {
            fail("Could not find a free port on 127.0.0.1.")
            return
        }
        guard let server = Server(resources: resources, port: port) else {
            fail("The bundled Python runtime is missing from \(resources.path).")
            return
        }
        self.server = server
        self.url = URL(string: "http://127.0.0.1:\(port)/")

        server.onEvent = { [weak self] in self?.handle($0) }
        server.onExit = { [weak self] status in self?.serverExited(status) }

        do {
            try server.start()
        } catch {
            fail("Could not start the server: \(error.localizedDescription)")
            return
        }

        // Generous: a cold first launch imports torch, opencv and mediapipe,
        // and on a slow connection it downloads 415 MB of weights first.
        waitForHealth(port: port, timeout: 1800) { [weak self] ok in
            guard let self, !self.quitting else { return }
            if ok { self.ready() } else {
                self.fail("The server did not answer /health. See ~/Library/Logs/Vitruve/vitruve.log.")
            }
        }
    }

    // MARK: events from the Python side

    private func handle(_ event: [String: Any]) {
        switch event["event"] as? String ?? "" {
        case "starting":
            window.headline.stringValue = "Starting Vitruve"
            window.detail.stringValue = "Checking model weights."
        case "weights_start":
            let total = (event["total_bytes"] as? NSNumber)?.doubleValue ?? 0
            window.headline.stringValue = "Downloading model weights"
            window.detail.stringValue =
                "First run only. \(bytes(total)) from the pinned sources, verified by sha256."
            window.bar.isIndeterminate = false
            window.bar.minValue = 0
            window.bar.maxValue = 1
            window.bar.doubleValue = 0
        case "weights_progress":
            let done = (event["done"] as? NSNumber)?.doubleValue ?? 0
            let total = (event["total"] as? NSNumber)?.doubleValue ?? 1
            window.bar.doubleValue = total > 0 ? done / total : 0
            window.detail.stringValue = "\(bytes(done)) of \(bytes(total))"
        case "weights_ok":
            window.bar.isIndeterminate = true
            window.bar.startAnimation(nil)
            window.headline.stringValue = "Starting the measurement engine"
            window.detail.stringValue = "Loading the models. This takes a few seconds."
        case "weights_failed":
            let messages = (event["messages"] as? [String])?.joined(separator: "; ") ?? "unknown"
            Log.shared.write("weights failed: " + messages)
            window.detail.stringValue = "Weights unavailable. The app will open; analysis will not run."
        case "serving":
            window.headline.stringValue = "Starting the measurement engine"
        case "fatal":
            fail((event["message"] as? String) ?? "The server failed to start.")
        default:
            break
        }
    }

    private func ready() {
        guard let url else { return }
        window.bar.stopAnimation(nil)
        window.bar.isHidden = true
        window.headline.stringValue = "Vitruve is running"
        window.detail.stringValue = url.absoluteString + "\nNothing leaves this Mac."
        window.openButton.isEnabled = true
        statusItem?.menu?.item(withTag: 1)?.title = "Running at " + url.absoluteString
        NSWorkspace.shared.open(url)
    }

    private func serverExited(_ status: Int32) {
        if quitting { return }
        let tail = server?.recentStderr.suffix(6).joined(separator: "\n") ?? ""
        fail("The Vitruve server stopped (exit \(status)).\n\(tail)")
    }

    private func fail(_ message: String) {
        window.bar.stopAnimation(nil)
        window.bar.isHidden = true
        window.headline.stringValue = "Vitruve could not start"
        window.detail.stringValue = message
        window.openButton.isEnabled = false
        Log.shared.write("fatal: " + message)
    }

    // MARK: chrome

    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        main.addItem(appItem)
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "About Vitruve", action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Open in Browser", action: #selector(openBrowser), keyEquivalent: "o")
        appMenu.addItem(withTitle: "Reveal Log in Finder", action: #selector(revealLog), keyEquivalent: "")
        appMenu.addItem(.separator())
        let quitItem = NSMenuItem(title: "Quit Vitruve", action: #selector(quit), keyEquivalent: "q")
        quitItem.target = self
        appMenu.addItem(quitItem)
        appItem.submenu = appMenu
        NSApp.mainMenu = main
        for item in appMenu.items where item.target == nil { item.target = self }
    }

    private func buildStatusItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        item.button?.image = NSImage(systemSymbolName: "square.circle", accessibilityDescription: "Vitruve")
        item.button?.image?.isTemplate = true
        let menu = NSMenu()
        let status = NSMenuItem(title: "Starting", action: nil, keyEquivalent: "")
        status.tag = 1
        status.isEnabled = false
        menu.addItem(status)
        menu.addItem(.separator())
        let open = NSMenuItem(title: "Open in Browser", action: #selector(openBrowser), keyEquivalent: "")
        open.target = self
        menu.addItem(open)
        let show = NSMenuItem(title: "Show Vitruve Window", action: #selector(showWindow), keyEquivalent: "")
        show.target = self
        menu.addItem(show)
        menu.addItem(.separator())
        let quitItem = NSMenuItem(title: "Quit Vitruve", action: #selector(quit), keyEquivalent: "")
        quitItem.target = self
        menu.addItem(quitItem)
        item.menu = menu
        statusItem = item
    }

    @objc private func openBrowser() {
        if let url { NSWorkspace.shared.open(url) }
    }

    @objc private func showWindow() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func revealLog() {
        let log = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/Vitruve/vitruve.log")
        NSWorkspace.shared.activateFileViewerSelecting([log])
    }

    @objc private func quit() {
        quitting = true
        NSApp.terminate(nil)
    }

    // Closing the only window quits, and quitting stops the server. The
    // alternative -- a window that closes and leaves a server running behind a
    // menu bar icon -- is how a non-technical user ends up with a face
    // analysis service they do not know is running.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ note: Notification) {
        quitting = true
        server?.stop()
    }
}

private func bytes(_ n: Double) -> String {
    if n >= 1e9 { return String(format: "%.2f GB", n / 1e9) }
    if n >= 1e6 { return String(format: "%.0f MB", n / 1e6) }
    return String(format: "%.0f kB", n / 1e3)
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.run()
