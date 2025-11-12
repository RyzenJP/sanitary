<?php
/**
 * Sensor Module Authentication Guard
 * Allows access to:
 * - Field workers
 * - Admins
 * - Verified clients (those approved by admin)
 */

// Start session if not already started
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

// Check if user is logged in
if (!isset($_SESSION['username']) || !isset($_SESSION['role'])) {
    header('Location: /sanitary/index.php');
    exit();
}

// Check user role and authorization
$role = $_SESSION['role'];
$allowed_roles = ['field_worker', 'admin'];

if (in_array($role, $allowed_roles)) {
    // Field workers and admins have full access
    $worker_id = $_SESSION['user_id'] ?? null;
    $username = $_SESSION['username'];
    $full_name = $_SESSION['full_name'] ?? $_SESSION['username'];
} elseif ($role === 'client') {
    // Clients need to be verified by admin to access water quality data
    require_once __DIR__ . '/../db.php';
    
    $client_id = $_SESSION['client_id'] ?? null;
    
    if (!$client_id) {
        header('Location: /sanitary/access_denied.php?reason=invalid_session&module=sensor');
        exit();
    }
    
    // Check if client is verified
    $stmt = $conn->prepare("SELECT verified FROM clients WHERE client_id = ?");
    $stmt->bind_param("i", $client_id);
    $stmt->execute();
    $result = $stmt->get_result();
    
    if ($result->num_rows === 0) {
        header('Location: /sanitary/access_denied.php?reason=client_not_found&module=sensor');
        exit();
    }
    
    $client = $result->fetch_assoc();
    $stmt->close();
    
    if ($client['verified'] != 1) {
        // Client is not verified by admin
        header('Location: /sanitary/access_denied.php?reason=not_verified&module=sensor');
        exit();
    }
    
    // Client is verified, set variables
    $username = $_SESSION['username'];
    $full_name = $_SESSION['name'] ?? $_SESSION['username'];
} else {
    // Unauthorized role
    header('Location: /sanitary/access_denied.php?reason=unauthorized_role&module=sensor');
    exit();
}

// User is authenticated and authorized
?>

