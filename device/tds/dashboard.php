<?php
session_start();

// Check if user is logged in and is a field worker
if (!isset($_SESSION['username']) || $_SESSION['role'] !== 'field_worker') {
    header('Location: ../../../index.php');
    exit();
}

$worker_id = $_SESSION['user_id'];
$username = $_SESSION['username'];
$role = $_SESSION['role'];
$full_name = $_SESSION['full_name'] ?? $_SESSION['username'];

// Database connection
$servername = "localhost";
$username = "u520834156_userBCWaters25";
$password = "^u4ctM3J8!w";
$dbname = "u520834156_DBBagoWaters25";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// Fetch water quality test results with client information - show both manual and automatic sensor submissions
$water_quality_query = "
    SELECT 
        wq.*,
        c.client_id,
        c.name as client_name,
        c.email as client_email,
        fw.name as field_worker_name,
        fw.username as field_worker_username,
        v.name as verified_by_name
    FROM water_quality wq
    LEFT JOIN clients c ON wq.client_id = c.client_id
    LEFT JOIN users fw ON wq.field_worker_id = fw.id
    LEFT JOIN users v ON wq.verified_by = v.id
    WHERE wq.field_worker_id = ? OR wq.field_worker_id = 1
    ORDER BY wq.recorded_at DESC
    LIMIT 50
";

$stmt = $conn->prepare($water_quality_query);
$stmt->bind_param("i", $worker_id);
$stmt->execute();
$water_quality_result = $stmt->get_result();
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>Water Quality Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.2.1"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Bootstrap Icons -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css" rel="stylesheet">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        :root {
            --primary-color: #3b82f6;
            --primary-dark: #2563eb;
            --secondary-color: #64748b;
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --light-bg: #f8fafc;
            --card-bg: #ffffff;
            --border-color: #e2e8f0;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #64748b;
            --shadow-sm: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            --border-radius: 12px;
            --border-radius-lg: 16px;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--light-bg);
            color: var(--text-primary);
            padding-top: 76px;
            min-height: 100vh;
        }


        .navbar {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        }


        .sidebar {
            width: 260px;
            background: var(--card-bg);
            border-right: 1px solid var(--border-color);
            height: calc(100vh - 76px);
            position: fixed;
            top: 76px;
            left: 0;
            transition: all 0.3s ease;
            overflow-y: auto;
            z-index: 1000;
        }

        .sidebar.collapsed {
            width: 70px;
        }

        .sidebar.collapsed .sidebar-link span {
            display: none;
        }

        .sidebar.collapsed .sidebar-link {
            justify-content: center;
            padding: 0.75rem;
        }

        .sidebar.collapsed .sidebar-link i {
            margin-right: 0;
            font-size: 1.5rem;
        }


        .sidebar-link {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            color: var(--text-secondary);
            text-decoration: none;
            border-radius: var(--border-radius);
            transition: all 0.3s ease;
            margin-bottom: 0.5rem;
        }

        .sidebar-link:hover,
        .sidebar-link.active {
            background-color: var(--primary-color);
            color: white;
        }

        .sidebar-link i {
            margin-right: 0.75rem;
            font-size: 1.25rem;
        }

        .main-content {
            flex: 1;
            padding: 2rem;
            transition: all 0.3s ease;
            margin-left: 260px;
        }

        .main-content.sidebar-collapsed {
            margin-left: 70px;
        }


        .main-container {
            display: flex;
            flex: 1;
        }

        .sidebar-toggle {
            background: none;
            border: none;
            color: white;
            font-size: 1.5rem;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: var(--border-radius);
            margin-right: 1rem;
            transition: all 0.3s ease;
        }

        .sidebar-toggle:hover {
            background-color: rgba(255, 255, 255, 0.1);
        }

        /* Ensure navbar elements are properly visible */
        .navbar-nav .btn {
            color: white;
            border-color: white;
        }

        .navbar-nav .btn:hover {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border-color: white;
        }

        .navbar-brand {
            color: white !important;
            font-weight: bold;
        }

        .navbar-nav span {
            color: white;
        }
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: #f5f7fa;
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 30px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            position: relative;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        
        .subtitle {
            font-size: 1.2rem;
            color: #7f8c8d;
            margin-bottom: 10px;
        }
        
        .info-button {
            position: absolute;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-weight: bold;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .spinning {
            animation: spin 1s linear infinite;
        }
        
        #manual-refresh-btn:hover {
            background: #2980b9;
        }
        
        #manual-refresh-btn:active {
            transform: scale(0.95);
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            position: relative;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .card-icon {
            font-size: 24px;
            margin-right: 10px;
            background: #e9ecef;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #2c3e50;
        }
        
        .card-title {
            font-size: 1.2rem;
            font-weight: 500;
            color: #2c3e50;
        }
        
        .value {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 10px 0;
            color: #2c3e50;
        }
        
        .unit {
            font-size: 1rem;
            color: #7f8c8d;
        }
        
        .chart-container {
            height: 300px;
            margin-top: 10px;
        }
        
        .status {
            display: flex;
            align-items: center;
            margin-top: 10px;
            font-size: 0.9rem;
            color: #7f8c8d;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .online {
            background-color: #27ae60;
        }
        
        .offline {
            background-color: #e74c3c;
        }
        
        .timestamp {
            font-size: 0.9rem;
            color: #95a5a6;
            margin-top: 5px;
        }
        
        .quality-indicator {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-top: 10px;
            background: #e9ecef;
            color: #2c3e50;
        }
        
        .good {
            background-color: #d5f5e3;
            color: #27ae60;
        }
        
        .moderate {
            background-color: #fdebd0;
            color: #f39c12;
        }
        
        .poor {
            background-color: #fadbd8;
            color: #e74c3c;
        }
        
        .history-title {
            margin: 30px 0 20px;
            font-size: 1.8rem;
            color: #2c3e50;
            padding-left: 10px;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 992px) {
            .charts-container {
                grid-template-columns: 1fr;
            }
        }
        
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 2rem;
            }
        }
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        
        .tab {
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: background 0.3s ease;
            font-weight: 500;
        }
        
        .tab.active {
            background: #3498db;
            color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .quality-scale {
            margin-top: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            font-size: 0.85rem;
        }
        
        .scale-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .scale-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: white;
            padding: 30px;
            border-radius: 15px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .close-modal {
            font-size: 1.5rem;
            cursor: pointer;
        }
        
        .quality-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        
        .quality-table th, .quality-table td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .quality-table th {
            background: #f8f9fa;
        }
        
        .legend {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            font-size: 0.9rem;
        }
        
        .legend-color {
            width: 15px;
            height: 15px;
            border-radius: 3px;
            margin-right: 5px;
        }
        
        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
        }
        
        .error-message {
            background-color: #fadbd8;
            color: #c0392b;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }

        /* Water Quality Table Styles */
        .wq-table-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid #e9ecef;
            margin-top: 30px;
        }

        .wq-table-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #e9ecef;
        }

        .wq-table-header h2 {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2c3e50;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .wq-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }

        .wq-table thead th {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            font-weight: 700;
            color: #1e293b;
            padding: 15px 12px;
            text-align: left;
            font-size: 0.8125rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #e2e8f0;
            white-space: nowrap;
        }

        .wq-table tbody td {
            padding: 15px 12px;
            border-bottom: 1px solid #e9ecef;
            vertical-align: middle;
            font-size: 0.9375rem;
        }

        .wq-table tbody tr {
            transition: all 0.3s ease;
        }

        .wq-table tbody tr:hover {
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            transform: scale(1.005);
        }

        .client-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            border: 1px solid #3b82f6;
            border-radius: 8px;
            font-weight: 600;
            color: #1e40af;
        }

        .client-id {
            background: #3b82f6;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .quality-badge {
            font-weight: 700;
            font-size: 0.875rem;
            padding: 6px 12px;
            border-radius: 8px;
            display: inline-block;
            text-align: center;
        }

        .quality-good {
            color: #155724;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border: 1px solid #28a745;
        }

        .quality-warning {
            color: #856404;
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 1px solid #ffc107;
        }

        .quality-danger {
            color: #721c24;
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            border: 1px solid #dc3545;
        }

        .status-badge {
            padding: 6px 14px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .status-safe {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 1px solid #28a745;
        }

        .status-unsafe {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 1px solid #dc3545;
        }

        .worker-info {
            font-size: 0.875rem;
        }

        .worker-info strong {
            color: #2c3e50;
            display: block;
        }

        .worker-info small {
            color: #7f8c8d;
            display: block;
            margin-top: 2px;
        }

        .client-name {
            font-weight: 600;
            color: #2c3e50;
        }

        .ai-recommendation {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            text-align: center;
            white-space: nowrap;
        }

        .ai-recommendation.good {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border: 1px solid #28a745;
        }

        .ai-recommendation.warning {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border: 1px solid #ffc107;
        }

        .ai-recommendation.danger {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border: 1px solid #dc3545;
        }

        .ai-recommendation i {
            font-size: 0.875rem;
        }

        .potability-score {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 700;
            text-align: center;
            white-space: nowrap;
            border: 2px solid;
        }

        .potability-score.score-excellent {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
            border-color: #28a745;
        }

        .potability-score.score-good {
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
            color: #0c5460;
            border-color: #17a2b8;
        }

        .potability-score.score-moderate {
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            color: #856404;
            border-color: #ffc107;
        }

        .potability-score.score-poor {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
            border-color: #dc3545;
        }

        .potability-score i {
            font-size: 0.875rem;
        }

        /* Remove horizontal scrollbar and make table fit */
        .table-responsive {
            overflow-x: hidden;
        }

        .wq-table {
            width: 100%;
            table-layout: fixed;
        }

        /* Column width adjustments */
        .wq-table th:nth-child(1), .wq-table td:nth-child(1) { width: 5%; } /* ID */
        .wq-table th:nth-child(2), .wq-table td:nth-child(2) { width: 15%; } /* CLIENT'S FULL NAME */
        .wq-table th:nth-child(3), .wq-table td:nth-child(3) { width: 12%; } /* LOCATION */
        .wq-table th:nth-child(4), .wq-table td:nth-child(4) { width: 8%; } /* TDS */
        .wq-table th:nth-child(5), .wq-table td:nth-child(5) { width: 8%; } /* TURBIDITY */
        .wq-table th:nth-child(6), .wq-table td:nth-child(6) { width: 12%; } /* POTABILITY SCORE */
        .wq-table th:nth-child(7), .wq-table td:nth-child(7) { width: 12%; } /* FIELDWORKER */
        .wq-table th:nth-child(8), .wq-table td:nth-child(8) { width: 13%; } /* RECORDED AT */
        .wq-table th:nth-child(9), .wq-table td:nth-child(9) { width: 15%; } /* AI RECOMMENDATION */

        /* AI Recommendation text wrapping */
        .ai-recommendation {
            font-size: 0.8rem;
            padding: 4px 8px;
            line-height: 1.3;
            white-space: normal;
            word-wrap: break-word;
            max-width: 100%;
        }

        /* Make other content fit better */
        .client-name {
            font-size: 0.9rem;
            word-wrap: break-word;
        }

        .worker-info {
            font-size: 0.8rem;
        }

        .worker-info strong {
            font-size: 0.8rem;
        }

        .quality-badge {
            font-size: 0.8rem;
            padding: 4px 8px;
        }

        .potability-score {
            font-size: 0.75rem;
            padding: 3px 6px;
            max-width: 100%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
        }

        .empty-state i {
            font-size: 4rem;
            color: #95a5a6;
            opacity: 0.5;
            margin-bottom: 20px;
        }

        .empty-state p {
            color: #7f8c8d;
            font-size: 1.1rem;
            margin: 0;
        }

        @media (max-width: 768px) {
            .wq-table {
                font-size: 0.85rem;
            }
            
            .wq-table thead th,
            .wq-table tbody td {
                padding: 10px 8px;
            }
        }
    </style>
</head>
<body>
    <!-- Navigation Bar -->
    <nav class="navbar navbar-expand-lg fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand text-white fw-bold" href="../../field_worker_dashboard.php">
                <i class="bi bi-hard-hat"></i> Field Worker Dashboard
            </a>
            <button class="sidebar-toggle" id="sidebarToggle">
                <i class="bi bi-list"></i>
            </button>
            <div class="navbar-nav ms-auto">
                <div class="d-flex align-items-center text-white">
                    <span class="me-3"><?php echo htmlspecialchars($full_name); ?></span>
                    <a href="#" class="btn btn-outline-light btn-sm" id="logoutBtn">
                        <i class="bi bi-box-arrow-right me-1"></i> Logout
                    </a>
                </div>
            </div>
        </div>
    </nav>

    <!-- Logout Confirmation Modal -->
    <div class="modal fade" id="logoutConfirmModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Confirm Logout</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    Are you sure you want to logout?
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <a href="../../logout.php" class="btn btn-primary">Logout</a>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Container -->
    <div class="main-container">
        <!-- Sidebar -->
        <div class="sidebar" id="sidebar">
            <ul class="list-unstyled p-3">
                <li>
                    <a href="../../../field_worker_dashboard.php" class="sidebar-link">
                        <i class="bi bi-speedometer2"></i>
                        <span>Dashboard</span>
                    </a>
                </li>
                <li>
                    <a href="../../../field_worker_location_map.php" class="sidebar-link">
                        <i class="bi bi-map"></i>
                        <span>Location Map</span>
                    </a>
                </li>
                <li>
                    <a href="../../../field_worker_my_appointments.php" class="sidebar-link">
                        <i class="bi bi-calendar-check"></i>
                        <span>My Appointments</span>
                    </a>
                </li>
                <li>
                    <a href="dashboard.php" class="sidebar-link active">
                        <i class="bi bi-activity"></i>
                        <span>Water Sensor</span>
                    </a>
                </li>
                <li>
                    <a href="../../../field_worker_water_potability_dashboard.php" class="sidebar-link">
                        <i class="bi bi-cpu"></i>
                        <span>AI Potability</span>
                    </a>
                </li>
                <li>
                    <a href="../../../field_worker_profile.php" class="sidebar-link">
                        <i class="bi bi-person"></i>
                        <span>My Profile</span>
                    </a>
                </li>
            </ul>
        </div>

        <!-- Main Content -->
        <div class="main-content">
    <div class="container">
        <header>
            <h1><i class="fas fa-tint"></i> Water Quality Monitoring</h1>
            <p class="subtitle">Real-time TDS and Turbidity analysis and visualization</p>
            <p class="subtitle">Device ID: <span id="device-id">SENSOR_001</span></p>
            <div class="subtitle">
                <span id="last-update-time" style="color: #7f8c8d; font-size: 0.9rem;">
                    <i class="fas fa-sync-alt" id="refresh-icon"></i> Last updated: Never
                </span>
                <button id="manual-refresh-btn" style="margin-left: 15px; padding: 5px 15px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer;">
                    <i class="fas fa-sync-alt"></i> Refresh Now
                </button>
            </div>
            <div class="info-button" id="info-button">i</div>
        </header>
        
        <div id="error-container" class="error-message" style="display: none;"></div>
        
        <div class="tabs">
            <div class="tab active" data-tab="tds">TDS Monitoring</div>
            <div class="tab" data-tab="turbidity">Turbidity Monitoring</div>
        </div>
        
        <div class="tab-content active" id="tds-tab">
            <div class="dashboard">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <h2 class="card-title">ADC Reading</h2>
                    </div>
                    <div class="value" id="adc-value">--</div>
                    <div class="status">
                        <div class="status-dot" id="adc-status-dot"></div>
                        <span>Last updated: <span id="adc-time">--</span></span>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <h2 class="card-title">Voltage</h2>
                    </div>
                    <div class="value" id="voltage-value">--<span class="unit">V</span></div>
                    <div class="status">
                        <div class="status-dot" id="voltage-status-dot"></div>
                        <span>Last updated: <span id="voltage-time">--</span></span>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">
                            <i class="fas fa-water"></i>
                        </div>
                        <h2 class="card-title">TDS Value</h2>
                    </div>
                    <div class="value" id="tds-value">--<span class="unit">ppm</span></div>
                    <div class="quality-indicator" id="quality-indicator">--</div>
                    <div class="quality-scale">
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #27ae60;"></div>
                            <span>Excellent (0-50 ppm)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #2ecc71;"></div>
                            <span>Good (50-150 ppm)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #f39c12;"></div>
                            <span>Acceptable (150-300 ppm)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #e67e22;"></div>
                            <span>Moderate (300-500 ppm)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #e74c3c;"></div>
                            <span>Poor (500-1200 ppm)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #c0392b;"></div>
                            <span>Very Bad (>1200 ppm)</span>
                        </div>
                    </div>
                    <div class="status">
                        <div class="status-dot" id="tds-status-dot"></div>
                        <span>Last updated: <span id="tds-time">--</span></span>
                    </div>
                </div>
            </div>
            
            <h2 class="history-title">
                <i class="fas fa-robot me-2"></i>AI Water Quality Analysis
            </h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <h5 class="text-center mb-3">
                            <i class="fas fa-chart-line me-2"></i>AI Potability Score Trend
                        </h5>
                        <div class="chart-container">
                            <canvas id="ai-potability-chart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <h5 class="text-center mb-3">
                            <i class="fas fa-shield-alt me-2"></i>Water Quality Status Distribution
                        </h5>
                        <div class="chart-container">
                            <canvas id="water-quality-status-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <h2 class="history-title mt-4">Sensor Data History</h2>
            <div class="card">
                <div class="chart-container">
                    <canvas id="tds-chart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="tab-content" id="turbidity-tab">
            <div class="dashboard">
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">
                            <i class="fas fa-vial"></i>
                        </div>
                        <h2 class="card-title">NTU Value</h2>
                    </div>
                    <div class="value" id="ntu-value">--<span class="unit">NTU</span></div>
                    <div class="quality-indicator" id="turbidity-quality-indicator">--</div>
                    <div class="quality-scale">
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #27ae60;"></div>
                            <span>Excellent (0.0-1.0 NTU)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #f39c12;"></div>
                            <span>Acceptable (1-5 NTU)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #e67e22;"></div>
                            <span>Concerning (>5 NTU)</span>
                        </div>
                        <div class="scale-item">
                            <div class="scale-color" style="background-color: #e74c3c;"></div>
                            <span>Unsafe (>50 NTU)</span>
                        </div>
                    </div>
                    <div class="status">
                        <div class="status-dot" id="ntu-status-dot"></div>
                        <span>Last updated: <span id="ntu-time">--</span></span>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <h2 class="card-title">Analog Value</h2>
                    </div>
                    <div class="value" id="analog-value">--</div>
                    <div class="status">
                        <div class="status-dot" id="analog-status-dot"></div>
                        <span>Last updated: <span id="analog-time">--</span></span>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <div class="card-icon">
                            <i class="fas fa-bolt"></i>
                        </div>
                        <h2 class="card-title">Sensor Voltage</h2>
                    </div>
                    <div class="value" id="sensor-voltage-value">--<span class="unit">V</span></div>
                    <div class="status">
                        <div class="status-dot" id="sensor-voltage-status-dot"></div>
                        <span>Last updated: <span id="sensor-voltage-time">--</span></span>
                    </div>
                </div>
            </div>
            
            <h2 class="history-title">
                <i class="fas fa-robot me-2"></i>AI Turbidity Analysis
            </h2>
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <h5 class="text-center mb-3">
                            <i class="fas fa-chart-area me-2"></i>Turbidity vs WHO Limit (AI-Monitored)
                        </h5>
                        <div class="chart-container">
                            <canvas id="ai-turbidity-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <h2 class="history-title mt-4">Sensor Data History</h2>
            <div class="charts-container">
                <div class="card">
                    <div class="chart-container">
                        <canvas id="ntu-chart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="chart-container">
                        <canvas id="voltage-chart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <!-- Water Quality Test Results Table -->
        <div class="wq-table-container">
            <div class="wq-table-header">
                <h2>
                    <i class="fas fa-table"></i>
                    My Water Quality Test Results
                </h2>
                <div class="d-flex align-items-center gap-2">
                    <span class="badge bg-info" style="font-size: 0.8rem; padding: 6px 12px;">
                        <i class="fas fa-user-hard-hat me-1"></i>
                        <?php echo htmlspecialchars($full_name); ?>
                    </span>
                    <span class="badge bg-primary" style="font-size: 0.9rem; padding: 8px 16px;">
                        <?php echo $water_quality_result->num_rows; ?> Records
                    </span>
                </div>
            </div>

            <div class="table-responsive">
                <table class="wq-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>CLIENT'S FULL NAME</th>
                            <th>LOCATION</th>
                            <th class="text-center">TDS</th>
                            <th class="text-center">TURBIDITY</th>
                            <th class="text-center">POTABILITY<br>SCORE</th>
                            <th>FIELDWORKER</th>
                            <th>RECORDED AT</th>
                            <th class="text-center">AI RECOMMENDATION</th>
                        </tr>
                    </thead>
                    <tbody>
                        <?php if ($water_quality_result->num_rows > 0): ?>
                            <?php while ($row = $water_quality_result->fetch_assoc()): ?>
                            <tr>
                                <td><strong>#<?php echo $row['id']; ?></strong></td>
                                <td>
                                    <div class="client-badge">
                                        <span class="client-id">ID: <?php echo $row['client_id']; ?></span>
                                        <span><?php echo htmlspecialchars($row['client_name'] ?? 'Unknown'); ?></span>
                                    </div>
                                </td>
                                <td>
                                    <i class="fas fa-map-marker-alt text-primary me-1"></i>
                                    <?php echo htmlspecialchars($row['location']); ?>
                                </td>
                                <td class="text-center">
                                    <span class="quality-badge <?php 
                                        // WHO standard: TDS <= 500 ppm is good (green), > 500 is bad (red)
                                        echo $row['tds_value'] <= 500 ? 'quality-good' : 'quality-danger'; 
                                    ?>">
                                        <?php echo number_format($row['tds_value'], 2); ?>
                                    </span>
                                </td>
                                <td class="text-center">
                                    <span class="quality-badge <?php 
                                        // WHO standard: Turbidity <= 1.0 NTU is good (green), > 1.0 is bad (red)
                                        echo $row['turbidity_value'] <= 1.0 ? 'quality-good' : 'quality-danger'; 
                                    ?>">
                                        <?php echo number_format($row['turbidity_value'], 2); ?>
                                    </span>
                                </td>
                                <td class="text-center">
                                    <span class="potability-score score-moderate" 
                                          data-test-id="<?php echo $row['id']; ?>-score">
                                        <i class="fas fa-spinner fa-spin"></i> ...
                                    </span>
                                </td>
                                <td>
                                    <div class="worker-info">
                                        <strong>
                                            <i class="fas fa-user-hard-hat me-1"></i>
                                            <?php echo htmlspecialchars($row['field_worker_name'] ?? 'N/A'); ?>
                                        </strong>
                                        <?php if ($row['field_worker_username']): ?>
                                        <small>@<?php echo htmlspecialchars($row['field_worker_username']); ?></small>
                                        <?php endif; ?>
                                    </div>
                                </td>
                                <td>
                                    <i class="fas fa-calendar text-muted me-1"></i>
                                    <?php echo date('M d, Y', strtotime($row['recorded_at'])); ?>
                                    <br>
                                    <small class="text-muted">
                                        <i class="fas fa-clock me-1"></i>
                                        <?php echo date('h:i A', strtotime($row['recorded_at'])); ?>
                                    </small>
                                </td>
                                <td>
                                    <span class="ai-recommendation warning" 
                                          data-test-id="<?php echo $row['id']; ?>"
                                          data-tds="<?php echo $row['tds_value']; ?>"
                                          data-turbidity="<?php echo $row['turbidity_value']; ?>"
                                          data-ph="<?php echo $row['ph_value'] ?? 7.0; ?>"
                                          data-temp="<?php echo $row['temperature'] ?? 25.0; ?>">
                                        <i class="fas fa-spinner fa-spin"></i> Loading AI...
                                    </span>
                                </td>
                            </tr>
                            <?php endwhile; ?>
                        <?php else: ?>
                            <tr>
                                <td colspan="9">
                                    <div class="empty-state">
                                        <i class="fas fa-inbox"></i>
                                        <p>No water quality test results found</p>
                                        <small class="text-muted">You haven't submitted any water quality test results yet. Submit your first test to see it here.</small>
                                    </div>
                                </td>
                            </tr>
                        <?php endif; ?>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Info Modal -->
    <div class="modal" id="info-modal">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Water Quality Standards</h2>
                <span class="close-modal">&times;</span>
            </div>
            
            <h3><i class="fas fa-vial"></i> Turbidity Sensor</h3>
            <p>Measures cloudiness of water, caused by particles (silt, microbes, dirt). Unit: NTU (Nephelometric Turbidity Unit).</p>
            
            <table class="quality-table">
                <tr>
                    <th>Range (NTU)</th>
                    <th>Water Quality</th>
                    <th>Remarks</th>
                </tr>
                <tr>
                    <td>0.0–1.0</td>
                    <td>Excellent</td>
                    <td>Very clear water (meets WHO standards for drinking)</td>
                </tr>
                <tr>
                    <td>1–5</td>
                    <td>Acceptable</td>
                    <td>Still acceptable for drinking, but higher side for safety</td>
                </tr>
                <tr>
                    <td>>5</td>
                    <td>Concerning</td>
                    <td>Not recommended for drinking (can hide pathogens, poor quality)</td>
                </tr>
                <tr>
                    <td>>50</td>
                    <td>Unsafe</td>
                    <td>Very turbid, unsafe, often polluted</td>
                </tr>
            </table>
            
            <h3><i class="fas fa-water"></i> TDS Sensor</h3>
            <p>Measures amount of dissolved salts, minerals, and impurities in water. Unit: ppm (parts per million).</p>
            
            <table class="quality-table">
                <tr>
                    <th>Range (ppm)</th>
                    <th>Water Quality</th>
                    <th>Remarks</th>
                </tr>
                <tr>
                    <td>0–50</td>
                    <td>Excellent</td>
                    <td>Very pure (distilled / RO water)</td>
                </tr>
                <tr>
                    <td>50–150</td>
                    <td>Good</td>
                    <td>Fresh, natural taste; ideal for drinking</td>
                </tr>
                <tr>
                    <td>150–300</td>
                    <td>Acceptable</td>
                    <td>Normal for groundwater/tap; still safe</td>
                </tr>
                <tr>
                    <td>300–500</td>
                    <td>Moderate</td>
                    <td>Upper WHO safe limit; may taste salty/bitter</td>
                </tr>
                <tr>
                    <td>500–1200</td>
                    <td>Poor</td>
                    <td>Not recommended for drinking; possible health risks</td>
                </tr>
                <tr>
                    <td>>1200</td>
                    <td>Very Bad</td>
                    <td>Unsafe for human consumption. Usually brackish or contaminated</td>
                </tr>
            </table>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #27ae60;"></div>
                    <span>Excellent</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #2ecc71;"></div>
                    <span>Good</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #f39c12;"></div>
                    <span>Acceptable</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e67e22;"></div>
                    <span>Moderate</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #e74c3c;"></div>
                    <span>Poor/Concerning</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #c0392b;"></div>
                    <span>Very Bad/Unsafe</span>
                </div>
            </div>
        </div>
    </div>
        </div> <!-- Close main-content -->
    </div> <!-- Close main-container -->

    <script>
        // API endpoints
        const API_BASE_URL = '/sanitary/main/sensor/device/tds/';
        const TDS_API_URL = `${API_BASE_URL}get_latest_data.php`;
        const TURBIDITY_API_URL = `${API_BASE_URL}get_turbidity_data.php`;
        
        // Default thresholds based on water quality standards
        const tdsThresholds = [
            { max: 50, quality: "Excellent", color: "#27ae60", class: "good" },
            { max: 150, quality: "Good", color: "#2ecc71", class: "good" },
            { max: 300, quality: "Acceptable", color: "#f39c12", class: "moderate" },
            { max: 500, quality: "Moderate", color: "#e67e22", class: "moderate" },
            { max: 1200, quality: "Poor", color: "#e74c3c", class: "poor" },
            { max: Infinity, quality: "Very Bad", color: "#c0392b", class: "poor" }
        ];
        
        const turbidityThresholds = [
            { max: 1.0, quality: "Excellent", color: "#27ae60", class: "good" },
            { max: 5.0, quality: "Acceptable", color: "#f39c12", class: "moderate" },
            { max: 50.0, quality: "Concerning", color: "#e67e22", class: "moderate" },
            { max: Infinity, quality: "Unsafe", color: "#e74c3c", class: "poor" }
        ];
        
        const deviceId = "SENSOR_001";

        // DOM elements
        const adcValue = document.getElementById('adc-value');
        const voltageValue = document.getElementById('voltage-value');
        const tdsValue = document.getElementById('tds-value');
        const adcTime = document.getElementById('adc-time');
        const voltageTime = document.getElementById('voltage-time');
        const tdsTime = document.getElementById('tds-time');
        const qualityIndicator = document.getElementById('quality-indicator');
        
        const ntuValue = document.getElementById('ntu-value');
        const analogValue = document.getElementById('analog-value');
        const sensorVoltageValue = document.getElementById('sensor-voltage-value');
        const ntuTime = document.getElementById('ntu-time');
        const analogTime = document.getElementById('analog-time');
        const sensorVoltageTime = document.getElementById('sensor-voltage-time');
        const turbidityQualityIndicator = document.getElementById('turbidity-quality-indicator');
        
        const infoButton = document.getElementById('info-button');
        const infoModal = document.getElementById('info-modal');
        const closeModal = document.querySelector('.close-modal');
        const errorContainer = document.getElementById('error-container');

        // Tab functionality
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const tabId = tab.getAttribute('data-tab');
                
                // Update active tab
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                // Show active content
                tabContents.forEach(content => content.classList.remove('active'));
                document.getElementById(`${tabId}-tab`).classList.add('active');
            });
        });

        // Modal functionality
        infoButton.addEventListener('click', () => {
            infoModal.style.display = 'flex';
        });
        
        closeModal.addEventListener('click', () => {
            infoModal.style.display = 'none';
        });
        
        window.addEventListener('click', (e) => {
            if (e.target === infoModal) {
                infoModal.style.display = 'none';
            }
        });

        // Show error message
        function showError(message) {
            errorContainer.textContent = message;
            errorContainer.style.display = 'block';
            setTimeout(() => {
                errorContainer.style.display = 'none';
            }, 5000);
        }

        // Update quality indicator based on TDS value
        function updateTdsQualityIndicator(tds) {
            for (const threshold of tdsThresholds) {
                if (tds <= threshold.max) {
                    qualityIndicator.textContent = threshold.quality;
                    qualityIndicator.className = `quality-indicator ${threshold.class}`;
                    qualityIndicator.style.backgroundColor = threshold.color;
                    qualityIndicator.style.color = 'white';
                    break;
                }
            }
        }

        // Update quality indicator based on NTU value
        function updateTurbidityQualityIndicator(ntu) {
            for (const threshold of turbidityThresholds) {
                if (ntu <= threshold.max) {
                    turbidityQualityIndicator.textContent = threshold.quality;
                    turbidityQualityIndicator.className = `quality-indicator ${threshold.class}`;
                    turbidityQualityIndicator.style.backgroundColor = threshold.color;
                    turbidityQualityIndicator.style.color = 'white';
                    break;
                }
            }
        }

        // Format date for display
        function formatDateTime(dateString) {
            const date = new Date(dateString);
            return date.toLocaleTimeString();
        }

        // Fetch TDS data from server
        async function fetchTdsData() {
            try {
                const response = await fetch(TDS_API_URL + '?t=' + Date.now(), {
                    cache: 'no-store',
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                }); // Add aggressive cache busting
                if (!response.ok) {
                    // Don't show error for 404 or empty data - just return null
                    if (response.status === 404) {
                        console.log('No TDS data available yet');
                        return null;
                    }
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                
                if (data.error) {
                    console.error('TDS API error:', data.error);
                    return null;
                }
                
                console.log('TDS data fetched:', data.readings ? data.readings.length + ' readings' : 'no readings');
                return data;
            } catch (error) {
                console.error('Error fetching TDS data:', error);
                // Only show error for actual network/server errors, not missing data
                if (error.message.includes('HTTP error! status: 404')) {
                    return null;
                }
                showError(`Failed to load TDS data: ${error.message}`);
                return null;
            }
        }

        // Fetch turbidity data from server
        async function fetchTurbidityData() {
            try {
                const response = await fetch(TURBIDITY_API_URL + '?t=' + Date.now(), {
                    cache: 'no-store',
                    headers: {
                        'Cache-Control': 'no-cache, no-store, must-revalidate',
                        'Pragma': 'no-cache',
                        'Expires': '0'
                    }
                }); // Add aggressive cache busting
                if (!response.ok) {
                    // Don't show error for 404 or empty data - just return null
                    if (response.status === 404) {
                        console.log('No turbidity data available yet');
                        return null;
                    }
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                
                if (data.error) {
                    console.error('Turbidity API error:', data.error);
                    return null;
                }
                
                console.log('Turbidity data fetched:', data.readings ? data.readings.length + ' readings' : 'no readings');
                return data;
            } catch (error) {
                console.error('Error fetching turbidity data:', error);
                // Only show error for actual network/server errors, not missing data
                if (error.message.includes('HTTP error! status: 404')) {
                    return null;
                }
                showError(`Failed to load turbidity data: ${error.message}`);
                return null;
            }
        }

        // Initialize charts with WHO standards visualization
        const tdsCtx = document.getElementById('tds-chart').getContext('2d');
        const tdsChart = new Chart(tdsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'TDS (ppm)',
                    data: [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: [],  // Will be set dynamically
                    pointBorderColor: [],      // Will be set dynamically
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#2c3e50'
                        }
                    },
                    annotation: {
                        annotations: {
                            whoLimit: {
                                type: 'line',
                                yMin: 500,
                                yMax: 500,
                                borderColor: 'rgba(239, 68, 68, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'WHO Limit (500 ppm)',
                                    enabled: true,
                                    position: 'end',
                                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                                    color: 'white',
                                    font: {
                                        weight: 'bold'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        const ntuCtx = document.getElementById('ntu-chart').getContext('2d');
        const ntuChart = new Chart(ntuCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Turbidity (NTU)',
                    data: [],
                    borderColor: '#9b59b6',
                    backgroundColor: 'rgba(155, 89, 182, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: [],  // Will be set dynamically
                    pointBorderColor: [],      // Will be set dynamically
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#2c3e50'
                        }
                    },
                    annotation: {
                        annotations: {
                            whoLimit: {
                                type: 'line',
                                yMin: 1.0,
                                yMax: 1.0,
                                borderColor: 'rgba(239, 68, 68, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'WHO Limit (1.0 NTU)',
                                    enabled: true,
                                    position: 'end',
                                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                                    color: 'white',
                                    font: {
                                        weight: 'bold'
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        const voltageCtx = document.getElementById('voltage-chart').getContext('2d');
        const voltageChart = new Chart(voltageCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Sensor Voltage (V)',
                    data: [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#e74c3c',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#2c3e50'
                        }
                    }
                }
            }
        });

        // Initialize AI-powered charts
        const aiPotabilityCtx = document.getElementById('ai-potability-chart').getContext('2d');
        const aiPotabilityChart = new Chart(aiPotabilityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'AI Potability Score (%)',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: [],
                    pointBorderColor: [],
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#2c3e50'
                        }
                    },
                    annotation: {
                        annotations: {
                            threshold: {
                                type: 'line',
                                yMin: 70,
                                yMax: 70,
                                borderColor: 'rgba(239, 68, 68, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'Potability Threshold (70%)',
                                    enabled: true,
                                    position: 'end',
                                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                                    color: 'white'
                                }
                            }
                        }
                    }
                }
            }
        });

        const waterQualityStatusCtx = document.getElementById('water-quality-status-chart').getContext('2d');
        const waterQualityStatusChart = new Chart(waterQualityStatusCtx, {
            type: 'doughnut',
            data: {
                labels: ['Potable', 'Not Potable'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(239, 68, 68, 0.8)'
                    ],
                    borderColor: [
                        '#10b981',
                        '#ef4444'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#2c3e50',
                            padding: 15,
                            font: {
                                size: 14
                            }
                        }
                    }
                }
            }
        });

        const aiTurbidityCtx = document.getElementById('ai-turbidity-chart').getContext('2d');
        const aiTurbidityChart = new Chart(aiTurbidityCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Turbidity (NTU)',
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#7f8c8d'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    annotation: {
                        annotations: {
                            threshold: {
                                type: 'line',
                                yMin: 1.0,
                                yMax: 1.0,
                                borderColor: 'rgba(239, 68, 68, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                label: {
                                    content: 'WHO Limit (1.0 NTU)',
                                    enabled: true,
                                    position: 'end',
                                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                                    color: 'white'
                                }
                            }
                        }
                    }
                }
            }
        });

        // Update the TDS dashboard with real data
        function updateTdsDashboard(data) {
            if (!data || !data.readings || data.readings.length === 0) {
                // Show "None" when no data is available
                adcValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
                voltageValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
                tdsValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
                
                adcTime.textContent = 'No Data';
                voltageTime.textContent = 'No Data';
                tdsTime.textContent = 'No Data';
                
                // Set status dots to offline (gray)
                document.getElementById('adc-status-dot').className = 'status-dot offline';
                document.getElementById('voltage-status-dot').className = 'status-dot offline';
                document.getElementById('tds-status-dot').className = 'status-dot offline';
                
                // Set quality indicator to default
                qualityIndicator.textContent = 'No Data';
                qualityIndicator.className = 'quality-indicator bg-secondary';
                qualityIndicator.style.backgroundColor = '#6c757d';
                qualityIndicator.style.color = 'white';
                
                // Clear chart data
                tdsChart.data.labels = [];
                tdsChart.data.datasets[0].data = [];
                tdsChart.update();
                
                return;
            }
            
            const latestReading = data.readings[0];
            
            // Update ADC value - show actual value or None
            if (latestReading.analog_value !== undefined && latestReading.analog_value !== null) {
                // Show actual value (0 or positive number)
                adcValue.textContent = latestReading.analog_value;
            } else {
                adcValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
            }
            
            // Update voltage value - show actual value or None
            if (latestReading.voltage !== undefined && latestReading.voltage !== null && !isNaN(latestReading.voltage)) {
                // Show actual value (0.000 or positive number)
                voltageValue.innerHTML = parseFloat(latestReading.voltage).toFixed(3) + '<span class="unit">V</span>';
            } else {
                voltageValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
            }
            
            // Update TDS value - show 0 for -1 (no water), actual value, or None
            if (latestReading.tds_value !== undefined && latestReading.tds_value !== null && !isNaN(latestReading.tds_value)) {
                if (parseFloat(latestReading.tds_value) === -1) {
                    // Show 0 instead of "No Water Detected"
                    tdsValue.innerHTML = '0.00<span class="unit">ppm</span>';
                } else {
                    tdsValue.innerHTML = parseFloat(latestReading.tds_value).toFixed(2) + '<span class="unit">ppm</span>';
                }
            } else {
                tdsValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
            }
            
            const timeString = formatDateTime(latestReading.reading_time);
            
            adcTime.textContent = timeString;
            voltageTime.textContent = timeString;
            tdsTime.textContent = timeString;
            
            // Set status dots to online (green)
            document.getElementById('adc-status-dot').className = 'status-dot online';
            document.getElementById('voltage-status-dot').className = 'status-dot online';
            document.getElementById('tds-status-dot').className = 'status-dot online';
            
            if (latestReading.tds_value !== undefined) {
                updateTdsQualityIndicator(latestReading.tds_value);
            }
            
            // Update chart with historical data and color-coded points
            const readings = data.readings.slice(0, 20).reverse(); // Get last 20 readings
            tdsChart.data.labels = readings.map(r => formatDateTime(r.reading_time));
            tdsChart.data.datasets[0].data = readings.map(r => r.tds_value);
            
            // Color-code points based on WHO limit (500 ppm)
            tdsChart.data.datasets[0].pointBackgroundColor = readings.map(r => 
                r.tds_value <= 500 ? '#10b981' : '#ef4444'  // Green for safe, red for unsafe
            );
            tdsChart.data.datasets[0].pointBorderColor = readings.map(r => 
                r.tds_value <= 500 ? '#10b981' : '#ef4444'
            );
            
            tdsChart.update();
        }

        // Update the turbidity dashboard with real data
        function updateTurbidityDashboard(data) {
            if (!data || !data.readings || data.readings.length === 0) {
                // Show "None" when no data is available
                ntuValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
                analogValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
                sensorVoltageValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
                
                ntuTime.textContent = 'No Data';
                analogTime.textContent = 'No Data';
                sensorVoltageTime.textContent = 'No Data';
                
                // Set status dots to offline (gray)
                document.getElementById('ntu-status-dot').className = 'status-dot offline';
                document.getElementById('analog-status-dot').className = 'status-dot offline';
                document.getElementById('sensor-voltage-status-dot').className = 'status-dot offline';
                
                // Set quality indicator to default
                turbidityQualityIndicator.textContent = 'No Data';
                turbidityQualityIndicator.className = 'quality-indicator bg-secondary';
                turbidityQualityIndicator.style.backgroundColor = '#6c757d';
                turbidityQualityIndicator.style.color = 'white';
                
                // Clear chart data
                ntuChart.data.labels = [];
                ntuChart.data.datasets[0].data = [];
                ntuChart.update();
                
                voltageChart.data.labels = [];
                voltageChart.data.datasets[0].data = [];
                voltageChart.update();
                
                return;
            }
            
            const latestReading = data.readings[0];
            
            // Update NTU value - show actual value or None
            if (latestReading.ntu_value !== undefined && latestReading.ntu_value !== null && !isNaN(latestReading.ntu_value)) {
                ntuValue.innerHTML = parseFloat(latestReading.ntu_value).toFixed(1) + '<span class="unit">NTU</span>';
            } else {
                ntuValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
            }
            
            // Update analog value - show actual value or None
            if (latestReading.analog_value !== undefined && latestReading.analog_value !== null) {
                analogValue.textContent = latestReading.analog_value;
            } else {
                analogValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
            }
            
            // Update sensor voltage - show actual value or None
            if (latestReading.sensor_voltage !== undefined && latestReading.sensor_voltage !== null && !isNaN(latestReading.sensor_voltage)) {
                sensorVoltageValue.innerHTML = parseFloat(latestReading.sensor_voltage).toFixed(2) + '<span class="unit">V</span>';
            } else {
                sensorVoltageValue.innerHTML = '<span style="font-size: 1.5rem; color: #7f8c8d;">None</span>';
            }
            
            const timeString = formatDateTime(latestReading.reading_time);
            
            ntuTime.textContent = timeString;
            analogTime.textContent = timeString;
            sensorVoltageTime.textContent = timeString;
            
            // Set status dots to online (green)
            document.getElementById('ntu-status-dot').className = 'status-dot online';
            document.getElementById('analog-status-dot').className = 'status-dot online';
            document.getElementById('sensor-voltage-status-dot').className = 'status-dot online';
            
            if (latestReading.ntu_value !== undefined) {
                updateTurbidityQualityIndicator(latestReading.ntu_value);
            }
            
            // Update charts with historical data and color-coded points
            const readings = data.readings.slice(0, 20).reverse(); // Get last 20 readings
            ntuChart.data.labels = readings.map(r => formatDateTime(r.reading_time));
            ntuChart.data.datasets[0].data = readings.map(r => r.ntu_value);
            
            // Color-code points based on WHO limit (1.0 NTU)
            ntuChart.data.datasets[0].pointBackgroundColor = readings.map(r => 
                r.ntu_value <= 1.0 ? '#10b981' : '#ef4444'  // Green for safe, red for unsafe
            );
            ntuChart.data.datasets[0].pointBorderColor = readings.map(r => 
                r.ntu_value <= 1.0 ? '#10b981' : '#ef4444'
            );
            
            ntuChart.update();
            
            voltageChart.data.labels = readings.map(r => formatDateTime(r.reading_time));
            voltageChart.data.datasets[0].data = readings.map(r => r.sensor_voltage);
            voltageChart.update();
        }

        // Update last refresh time
        function updateLastRefreshTime() {
            const lastUpdateElement = document.getElementById('last-update-time');
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            lastUpdateElement.innerHTML = `<i class="fas fa-sync-alt" id="refresh-icon"></i> Last updated: ${timeString}`;
        }

        // Update AI charts with data from water quality test results
        async function updateAICharts() {
            try {
                console.log('📊 Fetching recent water tests from database...');
                
                // Try to fetch directly from database for charts
                const response = await fetch(`../../../api/get_recent_water_tests.php?limit=10&t=${Date.now()}`);
                
                if (!response.ok) {
                    console.log('❌ Could not fetch water test data for charts (HTTP error)');
                    return extractFromTable();
                }
                
                const tests = await response.json();
                
                if (tests.error) {
                    console.log('❌ API error:', tests.error);
                    return extractFromTable();
                }
                
                console.log(`✅ Fetched ${tests.length} water tests from database`);
                
                if (tests && tests.length > 0) {
                    // Reverse to show oldest first (chronological order)
                    await processAICharts(tests.reverse());
                } else {
                    console.log('⚠️ No water test data found via API, trying table extraction...');
                    extractFromTable();
                }
            } catch (error) {
                console.error('❌ Error updating AI charts:', error);
                extractFromTable();
            }
        }
        
        // Fallback: Extract data from HTML table
        function extractFromTable() {
            console.log('🔄 Extracting data from HTML table as fallback...');
            const tbody = document.querySelector('.wq-table tbody');
            
            if (!tbody) {
                console.log('❌ Table not found');
                return;
            }
            
            const rows = tbody.querySelectorAll('tr');
            const tests = [];
            
            rows.forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length >= 5) {
                    const idText = cells[0].textContent.trim();
                    const id = parseInt(idText.replace('#', ''));
                    
                    // Extract just the numeric value from the badge
                    const tdsElement = cells[3].querySelector('.quality-badge');
                    const turbidityElement = cells[4].querySelector('.quality-badge');
                    
                    const tdsText = tdsElement ? tdsElement.textContent.trim() : cells[3].textContent.trim();
                    const turbidityText = turbidityElement ? turbidityElement.textContent.trim() : cells[4].textContent.trim();
                    
                    const tdsValue = parseFloat(tdsText);
                    const turbidityValue = parseFloat(turbidityText);
                    
                    console.log(`📊 Extracted Test #${id}: TDS=${tdsValue}, Turbidity=${turbidityValue}`);
                    
                    if (!isNaN(tdsValue) && !isNaN(turbidityValue)) {
                        tests.push({
                            id: id,
                            tds_value: tdsValue,
                            turbidity_value: turbidityValue,
                            ph_value: 7.0,
                            temperature: 25.0
                        });
                    }
                }
            });
            
            if (tests.length > 0) {
                console.log(`✅ Extracted ${tests.length} tests from table`);
                processAICharts(tests.slice(0, 10).reverse());
            } else {
                console.log('❌ No data extracted from table');
            }
        }
        
        // Process AI charts with test data
        async function processAICharts(tests) {
            try {
                console.log('🔧 Processing AI charts with test data:', tests);
                
                // Update AI Potability Score Trend Chart
                const labels = tests.map((test, index) => `Test #${test.id}`);
                const scores = [];
                const colors = [];
                
                // Calculate AI scores for each test
                console.log(`🤖 Calling ML server for ${tests.length} tests...`);
                for (const test of tests) {
                    console.log(`📤 Sending to ML: Test #${test.id}, TDS=${test.tds_value}, Turbidity=${test.turbidity_value}`);
                    
                    // Call ML server for each test
                    try {
                        const requestData = {
                            tds_value: parseFloat(test.tds_value),
                            turbidity_value: parseFloat(test.turbidity_value),
                            temperature: parseFloat(test.temperature) || 25,
                            ph_level: parseFloat(test.ph_value) || 7.0
                        };
                        
                        console.log(`📋 Request payload:`, requestData);
                        
                        // Use PHP proxy instead of direct ML server call (CORS issue)
                        const mlResponse = await fetch(`../../api/python_ml_server.php`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(requestData)
                        });
                        
                        console.log(`📥 ML Response status: ${mlResponse.status}`);
                        
                        if (mlResponse.ok) {
                            const mlData = await mlResponse.json();
                            console.log(`✅ Test #${test.id}: Score = ${mlData.potability_score}%`);
                            scores.push(mlData.potability_score);
                            colors.push(mlData.potability_score >= 70 ? '#10b981' : '#ef4444');
                        } else {
                            console.log(`❌ Test #${test.id}: ML server error (status ${mlResponse.status})`);
                            scores.push(0);
                            colors.push('#6c757d');
                        }
                    } catch (error) {
                        console.log(`❌ Test #${test.id}: ML server unreachable -`, error.message);
                        scores.push(0);
                        colors.push('#6c757d');
                    }
                }
                
                aiPotabilityChart.data.labels = labels;
                aiPotabilityChart.data.datasets[0].data = scores;
                aiPotabilityChart.data.datasets[0].pointBackgroundColor = colors;
                aiPotabilityChart.data.datasets[0].pointBorderColor = colors;
                aiPotabilityChart.update();
                
                // Update Water Quality Status Distribution Chart
                const potableCount = scores.filter(s => s >= 70).length;
                const notPotableCount = scores.filter(s => s < 70).length;
                waterQualityStatusChart.data.datasets[0].data = [potableCount, notPotableCount];
                waterQualityStatusChart.update();
                
                // Update AI Turbidity Chart
                const turbidityLabels = tests.map((test, index) => `#${test.id}`);
                const turbidityValues = tests.map(test => parseFloat(test.turbidity_value));
                const turbidityColors = turbidityValues.map(val => 
                    val <= 1.0 ? 'rgba(16, 185, 129, 0.7)' : 'rgba(239, 68, 68, 0.7)'
                );
                const turbidityBorders = turbidityValues.map(val => 
                    val <= 1.0 ? '#10b981' : '#ef4444'
                );
                
                aiTurbidityChart.data.labels = turbidityLabels;
                aiTurbidityChart.data.datasets[0].data = turbidityValues;
                aiTurbidityChart.data.datasets[0].backgroundColor = turbidityColors;
                aiTurbidityChart.data.datasets[0].borderColor = turbidityBorders;
                aiTurbidityChart.update();
                
                console.log('✅ AI charts updated successfully with', tests.length, 'tests');
            } catch (error) {
                console.error('Error processing AI charts:', error);
            }
        }

        // Load data from server
        async function loadData(isManual = false) {
            console.log('Loading data...' + (isManual ? ' (Manual refresh)' : ' (Auto refresh)'));
            
            // Show spinning animation
            const refreshIcon = document.getElementById('refresh-icon');
            if (refreshIcon) {
                refreshIcon.classList.add('spinning');
            }
            
            try {
                const tdsData = await fetchTdsData();
                updateTdsDashboard(tdsData); // Always update, even with null data
                
                const turbidityData = await fetchTurbidityData();
                updateTurbidityDashboard(turbidityData); // Always update, even with null data
                
                // Automatically submit sensor data to database
                await autoSubmitSensorReading(tdsData, turbidityData);
                
                // Update last refresh time
                updateLastRefreshTime();
                
                if (isManual) {
                    // Show brief success message for manual refresh
                    const btn = document.getElementById('manual-refresh-btn');
                    const originalText = btn.innerHTML;
                    btn.innerHTML = '<i class="fas fa-check"></i> Updated!';
                    btn.style.background = '#27ae60';
                    setTimeout(() => {
                        btn.innerHTML = originalText;
                        btn.style.background = '#3498db';
                    }, 2000);
                }
            } catch (error) {
                console.error('Error loading data:', error);
            } finally {
                // Stop spinning animation
                if (refreshIcon) {
                    refreshIcon.classList.remove('spinning');
                }
            }
        }

        // Track last submitted data to prevent duplicates (MUST BE DECLARED FIRST!)
        // Store in localStorage so it persists across page reloads
        let lastSubmittedData = null;
        try {
            const stored = localStorage.getItem('lastSubmittedSensorData');
            if (stored) {
                lastSubmittedData = JSON.parse(stored);
                console.log('📂 Loaded last submission from storage:', lastSubmittedData);
            }
        } catch (e) {
            console.log('⚠️ Could not load last submission data');
        }
        
        // Expose a function to clear submission history (for testing/debugging)
        window.clearSubmissionHistory = function() {
            localStorage.removeItem('lastSubmittedSensorData');
            lastSubmittedData = null;
            console.log('🗑️ Submission history cleared');
        };

        // Manual refresh button
        document.addEventListener('DOMContentLoaded', function() {
            const manualRefreshBtn = document.getElementById('manual-refresh-btn');
            if (manualRefreshBtn) {
                manualRefreshBtn.addEventListener('click', function() {
                    loadData(true);
                });
            }
        });

        // Initial load
        loadData();
        
        // Automatically submit sensor readings to database
        async function autoSubmitSensorReading(tdsData, turbidityData) {
            try {
                // Extract latest readings from the data
                if (!tdsData || !tdsData.readings || tdsData.readings.length === 0) {
                    console.log('⚠️ No TDS data to submit');
                    return;
                }
                
                if (!turbidityData || !turbidityData.readings || turbidityData.readings.length === 0) {
                    console.log('⚠️ No Turbidity data to submit');
                    return;
                }
                
                const latestTds = tdsData.readings[0];
                const latestTurbidity = turbidityData.readings[0];
                const tdsValue = parseFloat(latestTds.tds_value);
                const turbidityValue = parseFloat(latestTurbidity.ntu_value);
                const readingTimestamp = latestTds.reading_time; // Use reading timestamp
                
                console.log('🔍 Sensor values:', { tds: tdsValue, turbidity: turbidityValue, timestamp: readingTimestamp });
                
                // Check if this is the same data as last submission (prevent duplicates)
                if (lastSubmittedData && 
                    lastSubmittedData.tds === tdsValue && 
                    lastSubmittedData.turbidity === turbidityValue && 
                    lastSubmittedData.timestamp === readingTimestamp) {
                    console.log('⏭️ Skipping auto-submit - Same data already submitted');
                    console.log('📌 Last submission:', lastSubmittedData);
                    return;
                }
                
                // Additional safety check: Don't submit if we just submitted in the last 30 seconds
                if (lastSubmittedData && lastSubmittedData.submittedAt) {
                    const lastSubmitTime = new Date(lastSubmittedData.submittedAt);
                    const timeSinceLastSubmit = (new Date() - lastSubmitTime) / 1000; // seconds
                    
                    if (timeSinceLastSubmit < 30) {
                        console.log(`⏱️ Skipping auto-submit - Last submission was ${timeSinceLastSubmit.toFixed(0)}s ago (minimum 30s required)`);
                        return;
                    }
                }
                
                console.log('🆕 New data detected - will submit');
                console.log('📌 Last submission was:', lastSubmittedData);
                
                // Only submit if we have valid sensor data (both TDS and Turbidity must be positive, not -1)
                if (tdsValue > 0 && turbidityValue > 0) {
                    console.log('✅ Valid NEW data - Submitting to database:', { tds: tdsValue, turbidity: turbidityValue });
                    
                    const response = await fetch('../../../api/sensor_integration.php', {
                        method: 'POST',
                        credentials: 'include', // Important: Include session cookies
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            action: 'submit_field_reading',
                            client_id: 1, // Default client for sensor data
                            location: 'ESP32 Sensor Location',
                            tds_value: tdsValue,
                            turbidity_value: turbidityValue,
                            ph_value: 7.0, // Default pH
                            temperature: 25.0, // Default temperature
                            remarks: 'Automatic sensor reading from TDS dashboard'
                        })
                    });
                    
                    // Get response text first to check
                    const responseText = await response.text();
                    console.log('📤 API Response:', responseText.substring(0, 500)); // Show first 500 chars
                    
                    // Try to parse as JSON
                    try {
                        const result = JSON.parse(responseText);
                        if (result.success) {
                            console.log('✅ Sensor data automatically submitted to database');
                            console.log('📋 New record ID:', result.water_quality_id);
                            
                            // Save this submission to prevent future duplicates
                            lastSubmittedData = {
                                tds: tdsValue,
                                turbidity: turbidityValue,
                                timestamp: readingTimestamp,
                                recordId: result.water_quality_id,
                                submittedAt: new Date().toISOString()
                            };
                            
                            // Save to localStorage so it persists across page reloads
                            try {
                                localStorage.setItem('lastSubmittedSensorData', JSON.stringify(lastSubmittedData));
                                console.log('💾 Saved submission to localStorage:', lastSubmittedData);
                            } catch (e) {
                                console.log('⚠️ Could not save to localStorage');
                            }
                            
                            // Refresh the table to show new data (but lastSubmittedData will persist!)
                            setTimeout(() => {
                                location.reload();
                            }, 2000);
                        } else {
                            console.log('⚠️ Auto-submission failed:', result.error);
                        }
                    } catch (parseError) {
                        console.log('❌ API returned non-JSON response (likely PHP error)');
                        console.log('❌ Full response:', responseText);
                    }
                } else {
                    console.log('❌ Skipping auto-submit - Invalid data:', { tds: tdsValue, turbidity: turbidityValue, reason: tdsValue <= 0 ? 'TDS <= 0' : 'Turbidity <= 0' });
                }
            } catch (error) {
                console.log('⚠️ Auto-submission error:', error);
                console.log('⚠️ Error details:', error.message);
            }
        }

        // Set interval for updates (every 10 seconds for more responsive updates)
        setInterval(loadData, 10000);
        
        // Update AI charts every 30 seconds (less frequent since they're from database)
        setInterval(updateAICharts, 30000);
        
        console.log('Dashboard initialized - Auto-refresh every 10 seconds, AI charts every 30 seconds');

        // Load AI predictions for all table rows asynchronously
        async function loadAIPredictions() {
            const aiElements = document.querySelectorAll('.ai-recommendation[data-test-id]');
            
            console.log(`🤖 Loading AI predictions for ${aiElements.length} records in table...`);
            
            for (const element of aiElements) {
                const testId = element.getAttribute('data-test-id');
                const tds = parseFloat(element.getAttribute('data-tds'));
                const turbidity = parseFloat(element.getAttribute('data-turbidity'));
                const ph = parseFloat(element.getAttribute('data-ph'));
                const temp = parseFloat(element.getAttribute('data-temp'));
                
                console.log(`📋 Test #${testId}: TDS=${tds} ppm, Turbidity=${turbidity} NTU`);
                
                try {
                    // Call the same API that the admin modal uses
                    const response = await fetch(`../../../api/get_water_quality_details.php?test_id=${testId}`);
                    
                    if (response.ok) {
                        const result = await response.json();
                        
                        if (result.success) {
                            const data = result.data;
                            const recommendation = data.ai_recommendation;
                            const aiStatus = data.ai_status;
                            const aiScore = data.ai_potability_score;
                            
                            // Determine class based on recommendation text content (not just score)
                            let recommendationClass = 'danger'; // Default to red
                            
                            // Check if recommendation says water is POTABLE
                            if (recommendation.includes('POTABLE') && !recommendation.includes('NOT potable')) {
                                recommendationClass = 'good'; // Green
                            } else if (recommendation.includes('NOT potable') || 
                                       recommendation.includes('treatment required') ||
                                       recommendation.includes('High Turbidity') ||
                                       recommendation.includes('TDS exceeds')) {
                                recommendationClass = 'danger'; // Red
                            }
                            
                            // Update the AI recommendation element
                            element.className = `ai-recommendation ${recommendationClass}`;
                            element.innerHTML = recommendation;
                            
                            // Update the potability score element
                            const scoreElement = document.querySelector(`[data-test-id="${testId}-score"]`);
                            if (scoreElement) {
                                // Only GREEN or RED - no marginal (score >= 70% = Potable/Green, < 70% = Not Potable/Red)
                                let scoreClass = aiScore >= 70 ? 'score-excellent' : 'score-poor';
                                
                                scoreElement.className = `potability-score ${scoreClass}`;
                                scoreElement.innerHTML = `<i class="fas fa-percentage"></i> ${aiScore}%`;
                                console.log(`   💯 Potability Score: ${aiScore}% (${scoreClass === 'score-excellent' ? 'GREEN - Potable' : 'RED - Not Potable'})`);
                            } else {
                                console.log(`   ⚠️ Score element not found for test #${testId}`);
                            }
                            
                            console.log(`✅ Test #${testId} complete - Score: ${aiScore}%, Status: ${aiStatus}`);
                        } else {
                            throw new Error('API returned error: ' + result.error);
                        }
                    } else {
                        throw new Error('API not responding');
                    }
                } catch (error) {
                    console.log(`⚠️ AI prediction failed for test #${testId}, using fallback`);
                    
                    // Fallback to simple calculation
                    const fallbackResult = calculateFallbackPrediction(tds, turbidity);
                    element.className = `ai-recommendation ${fallbackResult.class}`;
                    element.innerHTML = fallbackResult.text;
                }
                
                // Small delay to avoid overwhelming the server
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            
            console.log('🎉 All AI predictions loaded!');
        }
        
        // Fallback calculation function
        function calculateFallbackPrediction(tds, turbidity) {
            let score = 100;
            
            // TDS scoring
            if (tds > 1200) {
                score -= 40;
            } else if (tds > 900) {
                score -= 30;
            } else if (tds > 600) {
                score -= 20;
            } else if (tds > 300) {
                score -= 10;
            }
            
            // Turbidity scoring
            if (turbidity > 25) {
                score -= 30;
            } else if (turbidity > 10) {
                score -= 20;
            } else if (turbidity > 5) {
                score -= 10;
            } else if (turbidity > 1) {
                score -= 5;
            }
            
            score = Math.max(0, Math.min(100, score));
            
            // Only Potable or Not Potable - no marginal (score >= 70% = Potable, < 70% = Not Potable)
            let text, className;
            if (score >= 70) {
                text = '✅ Water is POTABLE. No immediate action needed.';
                className = 'good';
            } else {
                text = '🔴 Water is NOT potable. Treatment required.';
                className = 'danger';
            }
            
            return { text, class: className };
        }
        
        // Load AI predictions when page is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                loadAIPredictions();
                // Update AI charts after predictions are loaded
                setTimeout(updateAICharts, 2000);
            });
        } else {
            loadAIPredictions();
            // Update AI charts after predictions are loaded
            setTimeout(updateAICharts, 2000);
        }

        // Sidebar toggle functionality
        document.addEventListener('DOMContentLoaded', function() {
            const sidebarToggle = document.getElementById('sidebarToggle');
            const sidebar = document.getElementById('sidebar');
            const mainContent = document.querySelector('.main-content');

            sidebarToggle.addEventListener('click', function() {
                sidebar.classList.toggle('collapsed');
                mainContent.classList.toggle('sidebar-collapsed');
            });

            // Logout button functionality
            const logoutBtn = document.getElementById('logoutBtn');
            if (logoutBtn) {
                logoutBtn.addEventListener('click', function(e) {
                    e.preventDefault();
                    const logoutModal = new bootstrap.Modal(document.getElementById('logoutConfirmModal'));
                    logoutModal.show();
                });
            }
        });
    </script>
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
<?php
$conn->close();
?>