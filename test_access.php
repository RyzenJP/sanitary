<?php
/**
 * Test page to verify sensor auth_check.php is working correctly
 */
include 'auth_check.php';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sensor Access Test - Success!</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
</head>
<body class="bg-light">
    <div class="container mt-5">
        <div class="card shadow-lg">
            <div class="card-header bg-success text-white">
                <h3><i class="bi bi-check-circle-fill me-2"></i> Access Test Successful!</h3>
            </div>
            <div class="card-body">
                <div class="alert alert-success">
                    <h4><i class="bi bi-shield-check me-2"></i> Authentication Passed!</h4>
                    <p class="mb-0">You have successfully accessed the sensor module.</p>
                </div>
                
                <h5 class="mt-4">Your Session Information:</h5>
                <table class="table table-bordered">
                    <tr>
                        <th width="200">User ID</th>
                        <td><?= htmlspecialchars($worker_id ?? 'N/A') ?></td>
                    </tr>
                    <tr>
                        <th>Username</th>
                        <td><?= htmlspecialchars($username ?? 'N/A') ?></td>
                    </tr>
                    <tr>
                        <th>Full Name</th>
                        <td><?= htmlspecialchars($full_name ?? 'N/A') ?></td>
                    </tr>
                    <tr>
                        <th>Role</th>
                        <td><span class="badge bg-primary"><?= htmlspecialchars($_SESSION['role'] ?? 'N/A') ?></span></td>
                    </tr>
                </table>
                
                <div class="alert alert-info mt-4">
                    <i class="bi bi-info-circle me-2"></i>
                    <strong>This means:</strong>
                    <ul class="mb-0 mt-2">
                        <li>The <code>sensor/auth_check.php</code> file is working correctly</li>
                        <li>You are logged in as a Field Worker</li>
                        <li>You have access to all sensor module pages</li>
                    </ul>
                </div>
                
                <div class="mt-4">
                    <a href="/sanitary/field_worker_dashboard.php" class="btn btn-primary">
                        <i class="bi bi-house-door me-2"></i> Back to Dashboard
                    </a>
                    <a href="/sanitary/sensor/device/tds/dashboard.php" class="btn btn-success">
                        <i class="bi bi-speedometer2 me-2"></i> TDS Dashboard
                    </a>
                    <a href="/sanitary/sensor/predictive_dashboard.php" class="btn btn-info">
                        <i class="bi bi-graph-up me-2"></i> Predictive Dashboard
                    </a>
                </div>
            </div>
        </div>
    </div>
</body>
</html>








