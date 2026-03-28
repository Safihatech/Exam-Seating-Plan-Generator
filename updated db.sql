-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Sep 26, 2025 at 04:25 AM
-- Server version: 10.4.32-MariaDB
-- PHP Version: 8.0.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `seating_plan_db`
--

DELIMITER $$
--
-- Procedures
--
CREATE DEFINER=`root`@`localhost` PROCEDURE `CleanOldActivityLogs` ()   BEGIN
    DELETE FROM activity_logs 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL 90 DAY);
    
    SELECT ROW_COUNT() as deleted_records, 'Old activity logs cleaned' as message;
END$$

CREATE DEFINER=`root`@`localhost` PROCEDURE `CleanOldLoginAttempts` ()   BEGIN
    DELETE FROM login_attempts 
    WHERE attempt_time < DATE_SUB(NOW(), INTERVAL 30 DAY);
    
    SELECT ROW_COUNT() as deleted_records, 'Old login attempts cleaned' as message;
END$$

CREATE DEFINER=`root`@`localhost` PROCEDURE `CleanOldPDFs` ()   BEGIN
    DECLARE retention_days INT DEFAULT 90;
    
    -- Get retention setting
    SELECT CAST(setting_value AS UNSIGNED) INTO retention_days
    FROM system_settings 
    WHERE setting_key = 'pdf_retention_days';
    
    -- Delete old PDF records
    DELETE FROM pdf_history 
    WHERE created_at < DATE_SUB(NOW(), INTERVAL retention_days DAY);
    
    SELECT ROW_COUNT() as deleted_records, 'Old PDF records cleaned' as message;
END$$

CREATE DEFINER=`root`@`localhost` PROCEDURE `GetUserStatistics` ()   BEGIN
    SELECT 
        COUNT(*) as total_users,
        COUNT(CASE WHEN role = 'admin' THEN 1 END) as admin_count,
        COUNT(CASE WHEN role = 'faculty' THEN 1 END) as faculty_count,
        COUNT(CASE WHEN role = 'student' THEN 1 END) as student_count,
        COUNT(CASE WHEN is_active = TRUE THEN 1 END) as active_users,
        COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) THEN 1 END) as new_users_this_month
    FROM users;
END$$

DELIMITER ;

-- --------------------------------------------------------

--
-- Stand-in structure for view `active_users_summary`
-- (See below for the actual view)
--
CREATE TABLE `active_users_summary` (
`role` enum('admin','faculty','student')
,`user_count` bigint(21)
,`active_count` bigint(21)
);

-- --------------------------------------------------------

--
-- Table structure for table `activity_logs`
--

CREATE TABLE `activity_logs` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `action` varchar(100) NOT NULL,
  `details` text DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `user_agent` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `activity_logs`
--

INSERT INTO `activity_logs` (`id`, `user_id`, `action`, `details`, `ip_address`, `user_agent`, `created_at`) VALUES
(1, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 12:37:28'),
(2, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 12:37:47'),
(3, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 12:38:32'),
(4, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 13:27:18'),
(5, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 13:27:40'),
(6, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 13:28:30'),
(7, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 13:34:33'),
(8, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 13:36:30'),
(9, 4, 'LOGIN', 'User fazil logged in', '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36', '2025-09-25 13:36:30'),
(10, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 13:36:49'),
(11, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 13:48:37'),
(12, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 13:57:46'),
(13, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 13:58:13'),
(14, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 13:58:37'),
(15, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 14:01:36'),
(16, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 14:01:39'),
(17, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 14:01:47'),
(18, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 14:01:50'),
(19, 4, 'LOGOUT', 'User fazil logged out', NULL, NULL, '2025-09-25 14:02:00'),
(20, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 18:12:35'),
(21, 4, 'LOGIN', 'User fazil logged in', '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36', '2025-09-25 18:12:35'),
(22, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 18:12:54'),
(23, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 18:16:03'),
(24, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:16:08'),
(25, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:16:24'),
(26, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:16:33'),
(27, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:16:44'),
(28, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:20:49'),
(29, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 18:21:35'),
(30, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:21:37'),
(31, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 18:21:54'),
(32, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:21:58'),
(33, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:23:13'),
(34, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:29:49'),
(35, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(36, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(37, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(38, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(39, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(40, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(41, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(42, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(43, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(44, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(45, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(46, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(47, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(48, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(49, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(50, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(51, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(52, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(53, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(54, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(55, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(56, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(57, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(58, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(59, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(60, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(61, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(62, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(63, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(64, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(65, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(66, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(67, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(68, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(69, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(70, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(71, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(72, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(73, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(74, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(75, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(76, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 18:30:17'),
(77, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 18:30:18'),
(78, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:30:24'),
(79, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:30:39'),
(80, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:38:37'),
(81, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:38:45'),
(82, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(83, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(84, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(85, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(86, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(87, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(88, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(89, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(90, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(91, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(92, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(93, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(94, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(95, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(96, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(97, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(98, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(99, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(100, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(101, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(102, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(103, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(104, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(105, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(106, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(107, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(108, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(109, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(110, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(111, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(112, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(113, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(114, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(115, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(116, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(117, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(118, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(119, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(120, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(121, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(122, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(123, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(124, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 18:39:32'),
(125, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 18:39:34'),
(126, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:39:38'),
(127, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:39:43'),
(128, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:39:50'),
(129, 4, 'LOGOUT', 'User fazil logged out', NULL, NULL, '2025-09-25 18:57:21'),
(130, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 18:57:27'),
(131, 4, 'LOGIN', 'User fazil logged in', '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36', '2025-09-25 18:57:27'),
(132, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 18:57:34'),
(133, 4, 'LOGOUT', 'User fazil logged out', NULL, NULL, '2025-09-25 18:58:18'),
(134, 4, 'LOGIN_SUCCESS', 'User fazil logged in successfully', NULL, NULL, '2025-09-25 19:01:49'),
(135, 4, 'LOGIN', 'User fazil logged in', '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36', '2025-09-25 19:01:49'),
(136, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:01:54'),
(137, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:01:59'),
(138, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(139, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(140, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(141, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(142, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(143, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(144, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(145, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(146, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(147, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(148, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(149, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(150, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(151, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(152, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(153, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(154, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(155, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(156, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(157, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(158, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(159, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(160, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(161, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(162, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(163, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(164, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(165, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(166, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(167, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(168, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(169, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(170, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(171, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(172, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(173, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(174, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(175, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(176, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(177, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(178, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(179, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(180, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-25 19:02:21'),
(181, 4, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-25 19:02:23'),
(182, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:02:27'),
(183, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:02:36'),
(184, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:03:36'),
(185, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:11:39'),
(186, 4, 'EMAIL_SENT', 'Seating plan sent to mfazii4563@gmail.com', NULL, NULL, '2025-09-25 19:11:48'),
(187, 4, 'EMAIL_SENT', 'Seating plan sent to mudinjakandupidi040101@gmail.com', NULL, NULL, '2025-09-25 19:11:48'),
(188, 4, 'EMAIL_SENT', 'Seating plan sent to mohamedfazil2584@gmail.com', NULL, NULL, '2025-09-25 19:11:48'),
(189, 4, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-25 19:11:48'),
(190, 6, 'LOGIN_SUCCESS', 'User safiha logged in successfully', NULL, NULL, '2025-09-26 01:49:13'),
(191, 6, 'LOGIN', 'User safiha logged in', '127.0.0.1', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36', '2025-09-26 01:49:13'),
(192, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(193, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(194, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(195, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(196, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(197, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(198, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(199, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(200, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(201, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(202, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(203, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(204, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(205, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(206, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(207, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(208, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(209, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(210, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(211, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(212, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(213, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(214, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(215, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(216, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(217, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(218, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(219, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(220, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(221, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(222, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(223, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(224, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(225, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(226, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(227, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(228, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(229, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(230, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(231, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(232, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(233, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(234, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 01:50:23'),
(235, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-26 01:50:25'),
(236, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 01:50:38'),
(237, 6, 'EMAIL_SENT', 'Seating plan sent to mfazii4563@gmail.com', NULL, NULL, '2025-09-26 01:51:07'),
(238, 6, 'EMAIL_SENT', 'Seating plan sent to mudinjakandupidi040101@gmail.com', NULL, NULL, '2025-09-26 01:51:07'),
(239, 6, 'EMAIL_SENT', 'Seating plan sent to mohamedfazil2584@gmail.com', NULL, NULL, '2025-09-26 01:51:07'),
(240, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 01:51:07'),
(241, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 01:54:46'),
(242, NULL, 'EMAIL_ERROR', 'SMTP Auth error for mudinjakandupidi040101@gmail.com: SMTP Authentication failed. Please check email credentials: (535, b\'5.7.8 Username and Password not accepted. For more information, go to\\n5.7.8  https://support.google.com/mail/?p=BadCredentials d9443c01a7336-27ed6882160sm37196225ad.71 - gsmtp\')', NULL, NULL, '2025-09-26 01:54:56'),
(243, NULL, 'EMAIL_ERROR', 'SMTP Auth error for mohamedfazil2584@gmail.com: SMTP Authentication failed. Please check email credentials: (535, b\'5.7.8 Username and Password not accepted. For more information, go to\\n5.7.8  https://support.google.com/mail/?p=BadCredentials 41be03b00d2f7-b57c53b97a7sm3298366a12.8 - gsmtp\')', NULL, NULL, '2025-09-26 01:54:58'),
(244, NULL, 'EMAIL_ERROR', 'SMTP Auth error for mfazii4563@gmail.com: SMTP Authentication failed. Please check email credentials: (535, b\'5.7.8 Username and Password not accepted. For more information, go to\\n5.7.8  https://support.google.com/mail/?p=BadCredentials d9443c01a7336-27ed69bc273sm37429685ad.124 - gsmtp\')', NULL, NULL, '2025-09-26 01:55:00'),
(245, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 01:55:00'),
(246, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:02:20'),
(247, NULL, 'EMAIL_SENT', 'Email sent successfully to mfazii4563@gmail.com with PDF attachment (seating_plan_20250926_072031.pdf)', NULL, NULL, '2025-09-26 02:02:29'),
(248, 6, 'EMAIL_SENT', 'Seating plan sent to mfazii4563@gmail.com', NULL, NULL, '2025-09-26 02:02:29'),
(249, NULL, 'EMAIL_SENT', 'Email sent successfully to mudinjakandupidi040101@gmail.com with PDF attachment (seating_plan_20250926_072031.pdf)', NULL, NULL, '2025-09-26 02:02:33'),
(250, 6, 'EMAIL_SENT', 'Seating plan sent to mudinjakandupidi040101@gmail.com', NULL, NULL, '2025-09-26 02:02:33'),
(251, NULL, 'EMAIL_SENT', 'Email sent successfully to mohamedfazil2584@gmail.com with PDF attachment (seating_plan_20250926_072031.pdf)', NULL, NULL, '2025-09-26 02:02:36'),
(252, 6, 'EMAIL_SENT', 'Seating plan sent to mohamedfazil2584@gmail.com', NULL, NULL, '2025-09-26 02:02:36'),
(253, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:02:36'),
(254, NULL, 'EMAIL_SENT', 'Email sent successfully to mfazii4563@gmail.com with PDF attachment (seating_plan_20250926_072031.pdf)', NULL, NULL, '2025-09-26 02:03:47'),
(255, 6, 'EMAIL_SENT', 'Seating plan sent to mfazii4563@gmail.com', NULL, NULL, '2025-09-26 02:03:47'),
(256, NULL, 'EMAIL_SENT', 'Email sent successfully to mohamedfazil2584@gmail.com with PDF attachment (seating_plan_20250926_072031.pdf)', NULL, NULL, '2025-09-26 02:03:50'),
(257, 6, 'EMAIL_SENT', 'Seating plan sent to mohamedfazil2584@gmail.com', NULL, NULL, '2025-09-26 02:03:50'),
(258, NULL, 'EMAIL_SENT', 'Email sent successfully to mudinjakandupidi040101@gmail.com with PDF attachment (seating_plan_20250926_072031.pdf)', NULL, NULL, '2025-09-26 02:03:53'),
(259, 6, 'EMAIL_SENT', 'Seating plan sent to mudinjakandupidi040101@gmail.com', NULL, NULL, '2025-09-26 02:03:53'),
(260, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:03:54'),
(261, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:04:28'),
(262, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(263, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(264, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(265, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(266, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(267, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(268, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(269, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(270, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(271, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(272, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(273, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(274, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(275, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(276, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(277, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(278, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(279, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(280, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(281, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(282, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(283, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(284, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(285, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(286, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(287, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(288, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(289, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(290, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(291, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(292, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(293, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(294, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(295, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(296, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(297, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(298, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(299, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(300, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(301, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(302, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(303, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(304, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 02:05:03'),
(305, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-26 02:05:03'),
(306, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:05:13'),
(307, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:05:44'),
(308, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(309, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(310, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(311, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(312, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(313, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(314, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(315, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(316, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(317, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(318, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(319, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(320, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(321, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(322, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(323, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(324, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(325, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(326, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(327, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(328, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(329, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(330, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(331, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(332, NULL, 'STUDENT_UPDATED', 'Student 229X1A2908 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(333, NULL, 'STUDENT_UPDATED', 'Student 229X1A2909 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(334, NULL, 'STUDENT_UPDATED', 'Student 229X1A2910 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(335, NULL, 'STUDENT_UPDATED', 'Student 229X1A2912 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(336, NULL, 'STUDENT_UPDATED', 'Student 229X1A2915 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(337, NULL, 'STUDENT_UPDATED', 'Student 229X1A2916 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(338, NULL, 'STUDENT_UPDATED', 'Student 229X1A2917 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(339, NULL, 'STUDENT_UPDATED', 'Student 229X1A2918 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(340, NULL, 'STUDENT_UPDATED', 'Student 229X1A2920 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(341, NULL, 'STUDENT_UPDATED', 'Student 229X1A2922 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(342, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C3 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(343, NULL, 'STUDENT_UPDATED', 'Student 239X5A02C5 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(344, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D0 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(345, NULL, 'STUDENT_UPDATED', 'Student 239X5A02D6 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(346, NULL, 'STUDENT_UPDATED', 'Student 229X1A2901 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(347, NULL, 'STUDENT_UPDATED', 'Student 229X1A2902 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(348, NULL, 'STUDENT_UPDATED', 'Student 229X1A2905 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(349, NULL, 'STUDENT_UPDATED', 'Student 229X1A2906 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(350, NULL, 'STUDENT_UPDATED', 'Student 229X1A2907 () updated', NULL, NULL, '2025-09-26 02:06:01'),
(351, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 48 students in 3 rooms', NULL, NULL, '2025-09-26 02:06:02'),
(352, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:06:11'),
(353, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 24 students in 3 rooms', NULL, NULL, '2025-09-26 02:10:53'),
(354, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:11:31'),
(355, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 24 students in 3 rooms', NULL, NULL, '2025-09-26 02:12:08'),
(356, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:12:34'),
(357, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 114 students in 8 rooms', NULL, NULL, '2025-09-26 02:20:22'),
(358, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 90 students in 6 rooms', NULL, NULL, '2025-09-26 02:23:07'),
(359, 6, 'SEATING_PLAN_GENERATED', 'Generated seating plan with 90 students in 6 rooms', NULL, NULL, '2025-09-26 02:23:27'),
(360, 6, 'PDF_HISTORY_VIEW', 'User viewed PDF history', NULL, NULL, '2025-09-26 02:23:45');

-- --------------------------------------------------------

--
-- Table structure for table `email_notifications`
--

CREATE TABLE `email_notifications` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `recipient_email` varchar(100) NOT NULL,
  `subject` varchar(255) NOT NULL,
  `body` text NOT NULL,
  `sent` tinyint(1) DEFAULT 0,
  `sent_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `login_attempts`
--

CREATE TABLE `login_attempts` (
  `id` int(11) NOT NULL,
  `username` varchar(50) DEFAULT NULL,
  `ip_address` varchar(45) DEFAULT NULL,
  `success` tinyint(1) NOT NULL,
  `attempt_time` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `password_reset_tokens`
--

CREATE TABLE `password_reset_tokens` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `token` varchar(255) NOT NULL,
  `expires_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp(),
  `used` tinyint(1) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `pdf_history`
--

CREATE TABLE `pdf_history` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `filename` varchar(255) NOT NULL,
  `pdf_path` varchar(500) DEFAULT NULL,
  `file_path` varchar(500) NOT NULL,
  `student_count` int(11) NOT NULL,
  `room_count` int(11) NOT NULL,
  `students_per_desk` int(11) NOT NULL,
  `include_detained` tinyint(1) NOT NULL DEFAULT 0,
  `building` varchar(100) DEFAULT NULL,
  `plan_details` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `pdf_history`
--

INSERT INTO `pdf_history` (`id`, `user_id`, `filename`, `pdf_path`, `file_path`, `student_count`, `room_count`, `students_per_desk`, `include_detained`, `building`, `plan_details`, `created_at`) VALUES
(1, 4, 'seating_plan_20250925_234259.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250925_234259.pdf', '', 48, 3, 1, 0, 'Main Building', NULL, '2025-09-25 18:13:00'),
(2, 4, 'seating_plan_20250925_234605.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250925_234605.pdf', '', 48, 3, 1, 0, 'Main Building', NULL, '2025-09-25 18:16:06'),
(3, 4, 'seating_plan_20250925_235156.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250925_235156.pdf', '', 48, 3, 2, 0, 'Main Building', NULL, '2025-09-25 18:21:57'),
(4, 4, 'seating_plan_20250926_000021.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_000021.pdf', '', 48, 3, 2, 0, 'Back Side Building', NULL, '2025-09-25 18:30:22'),
(5, 4, 'seating_plan_20250926_000937.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_000937.pdf', '', 48, 3, 2, 0, 'basement ', NULL, '2025-09-25 18:39:38'),
(6, 4, 'seating_plan_20250926_003225.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_003225.pdf', '', 48, 3, 1, 0, 'Main Building', NULL, '2025-09-25 19:02:25'),
(7, 6, 'seating_plan_20250926_072031.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_072031.pdf', '', 48, 3, 2, 0, 'A Block', NULL, '2025-09-26 01:50:32'),
(8, 6, 'seating_plan_20250926_073512.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_073512.pdf', '', 48, 3, 1, 0, 'side Building', NULL, '2025-09-26 02:05:12'),
(9, 6, 'seating_plan_20250926_073608.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_073608.pdf', '', 48, 3, 2, 1, 'Main Building', NULL, '2025-09-26 02:06:09'),
(10, 6, 'seating_plan_20250926_074129.pdf', 'C:\\CODE_BASE\\Flutter Apps\\learnings\\sabi-s_project\\data\\pdfs\\seating_plan_20250926_074129.pdf', '', 24, 3, 2, 1, 'C BLOCK', NULL, '2025-09-26 02:11:30');

-- --------------------------------------------------------

--
-- Stand-in structure for view `pdf_statistics`
-- (See below for the actual view)
--
CREATE TABLE `pdf_statistics` (
`generation_date` date
,`total_pdfs` bigint(21)
,`total_students` decimal(32,0)
,`total_rooms` decimal(32,0)
,`avg_students_per_desk` decimal(14,4)
,`building` varchar(100)
);

-- --------------------------------------------------------

--
-- Stand-in structure for view `recent_activity`
-- (See below for the actual view)
--
CREATE TABLE `recent_activity` (
`id` int(11)
,`username` varchar(50)
,`full_name` varchar(100)
,`action` varchar(100)
,`details` text
,`ip_address` varchar(45)
,`created_at` timestamp
);

-- --------------------------------------------------------

--
-- Table structure for table `rooms_config`
--

CREATE TABLE `rooms_config` (
  `id` int(11) NOT NULL,
  `room_number` varchar(50) NOT NULL,
  `building` varchar(100) DEFAULT NULL,
  `room_rows` int(11) NOT NULL,
  `room_columns` int(11) NOT NULL,
  `capacity` int(11) GENERATED ALWAYS AS (`room_rows` * `room_columns`) STORED,
  `allowed_branches` text DEFAULT NULL,
  `is_active` tinyint(1) DEFAULT 1,
  `created_by` int(11) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `settings`
--

CREATE TABLE `settings` (
  `id` int(11) NOT NULL,
  `setting_key` varchar(100) NOT NULL,
  `setting_value` text DEFAULT NULL,
  `updated_by` int(11) DEFAULT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `students`
--

CREATE TABLE `students` (
  `id` int(11) NOT NULL,
  `roll_number` varchar(50) NOT NULL,
  `name` varchar(100) NOT NULL,
  `branch` varchar(50) NOT NULL,
  `section` varchar(10) DEFAULT NULL,
  `is_detained` tinyint(1) DEFAULT 0,
  `email` varchar(100) DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `students`
--

INSERT INTO `students` (`id`, `roll_number`, `name`, `branch`, `section`, `is_detained`, `email`, `created_at`) VALUES
(1, '229X1A2901', '', '', '', 0, 'mohamedfazil2584@gmail.com', '2025-09-25 12:37:27'),
(2, '229X1A2902', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(3, '229X1A2905', '', '', '', 0, 'mfazii4563@gmail.com', '2025-09-25 12:37:27'),
(4, '229X1A2906', '', '', '', 0, 'mohamedfazil2584@gmail.com', '2025-09-25 12:37:27'),
(5, '229X1A2907', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(6, '229X1A2908', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(7, '229X1A2909', '', '', '', 0, 'mfazii4563@gmail.com', '2025-09-25 12:37:27'),
(8, '229X1A2910', '', '', '', 0, 'mohamedfazil2584@gmail.com', '2025-09-25 12:37:27'),
(9, '229X1A2912', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(10, '229X1A2915', '', '', '', 0, 'mfazii4563@gmail.com', '2025-09-25 12:37:27'),
(11, '229X1A2916', '', '', '', 0, 'mohamedfazil2584@gmail.com', '2025-09-25 12:37:27'),
(12, '229X1A2917', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(13, '229X1A2918', '', '', '', 0, 'mfazii4563@gmail.com', '2025-09-25 12:37:27'),
(14, '229X1A2920', '', '', '', 0, 'mohamedfazil2584@gmail.com', '2025-09-25 12:37:27'),
(15, '229X1A2922', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(16, '239X5A02C3', '', '', '', 0, 'mfazii4563@gmail.com', '2025-09-25 12:37:27'),
(17, '239X5A02C5', '', '', '', 0, 'mohamedfazil2584@gmail.com', '2025-09-25 12:37:27'),
(18, '239X5A02D0', '', '', '', 0, 'mudinjakandupidi040101@gmail.com', '2025-09-25 12:37:27'),
(19, '239X5A02D6', '', '', '', 0, 'mfazii4563@gmail.com', '2025-09-25 12:37:27');

--
-- Triggers `students`
--
DELIMITER $$
CREATE TRIGGER `student_update_audit` AFTER UPDATE ON `students` FOR EACH ROW BEGIN
    INSERT INTO activity_logs (user_id, action, details, created_at)
    VALUES (NULL, 'STUDENT_UPDATED', 
           CONCAT('Student ', NEW.roll_number, ' (', NEW.name, ') updated'), NOW());
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Table structure for table `system_settings`
--

CREATE TABLE `system_settings` (
  `id` int(11) NOT NULL,
  `setting_key` varchar(100) NOT NULL,
  `setting_value` text DEFAULT NULL,
  `description` text DEFAULT NULL,
  `updated_by` int(11) DEFAULT NULL,
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `full_name` varchar(100) NOT NULL,
  `role` enum('admin','faculty','student') DEFAULT 'student',
  `is_active` tinyint(1) DEFAULT 1,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `updated_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `email`, `password_hash`, `full_name`, `role`, `is_active`, `created_at`, `updated_at`) VALUES
(1, 'admin', 'admin@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LsRyZy6DYgI2rk9jK', 'System Administrator', 'admin', 1, '2025-09-25 10:44:16', '2025-09-25 10:44:16'),
(4, 'fazil', 'fazil@gmail.com', '$2b$12$ZQf2vI55K2Xp/5NWn2/nP.dcRAjXCidBi07QgJGsVEqcE4YxCXbUe', 'fazil', 'admin', 1, '2025-09-25 11:59:36', '2025-09-25 11:59:36'),
(6, 'safiha', 'safiha@gmail.com', '$2b$12$IoXlXuZ8qzuhSrhL3ZHu2eVNvM.IKjHelH.tTF3gbAk4ugzyOx.Eq', 'safiha', 'admin', 1, '2025-09-26 01:49:02', '2025-09-26 01:49:02');

--
-- Triggers `users`
--
DELIMITER $$
CREATE TRIGGER `user_update_audit` AFTER UPDATE ON `users` FOR EACH ROW BEGIN
    INSERT INTO activity_logs (user_id, action, details, created_at)
    VALUES (NEW.id, 'USER_UPDATED', 
           CONCAT('User profile updated: ', 
                 CASE 
                   WHEN OLD.username != NEW.username THEN CONCAT('username changed from ', OLD.username, ' to ', NEW.username, '; ')
                   ELSE ''
                 END,
                 CASE 
                   WHEN OLD.email != NEW.email THEN CONCAT('email changed from ', OLD.email, ' to ', NEW.email, '; ')
                   ELSE ''
                 END,
                 CASE 
                   WHEN OLD.role != NEW.role THEN CONCAT('role changed from ', OLD.role, ' to ', NEW.role, '; ')
                   ELSE ''
                 END,
                 CASE 
                   WHEN OLD.is_active != NEW.is_active THEN CONCAT('active status changed to ', NEW.is_active, '; ')
                   ELSE ''
                 END
           ), NOW());
END
$$
DELIMITER ;

-- --------------------------------------------------------

--
-- Stand-in structure for view `user_dashboard_stats`
-- (See below for the actual view)
--
CREATE TABLE `user_dashboard_stats` (
`user_id` int(11)
,`username` varchar(50)
,`role` enum('admin','faculty','student')
,`total_pdfs_generated` bigint(21)
,`last_pdf_generated` timestamp
,`total_activities` bigint(21)
);

-- --------------------------------------------------------

--
-- Structure for view `active_users_summary`
--
DROP TABLE IF EXISTS `active_users_summary`;

CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `active_users_summary`  AS SELECT `users`.`role` AS `role`, count(0) AS `user_count`, count(case when `users`.`is_active` = 1 then 1 end) AS `active_count` FROM `users` GROUP BY `users`.`role` ;

-- --------------------------------------------------------

--
-- Structure for view `pdf_statistics`
--
DROP TABLE IF EXISTS `pdf_statistics`;

CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `pdf_statistics`  AS SELECT cast(`pdf_history`.`created_at` as date) AS `generation_date`, count(0) AS `total_pdfs`, sum(`pdf_history`.`student_count`) AS `total_students`, sum(`pdf_history`.`room_count`) AS `total_rooms`, avg(`pdf_history`.`students_per_desk`) AS `avg_students_per_desk`, `pdf_history`.`building` AS `building` FROM `pdf_history` GROUP BY cast(`pdf_history`.`created_at` as date), `pdf_history`.`building` ORDER BY cast(`pdf_history`.`created_at` as date) DESC ;

-- --------------------------------------------------------

--
-- Structure for view `recent_activity`
--
DROP TABLE IF EXISTS `recent_activity`;

CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `recent_activity`  AS SELECT `al`.`id` AS `id`, `u`.`username` AS `username`, `u`.`full_name` AS `full_name`, `al`.`action` AS `action`, `al`.`details` AS `details`, `al`.`ip_address` AS `ip_address`, `al`.`created_at` AS `created_at` FROM (`activity_logs` `al` left join `users` `u` on(`al`.`user_id` = `u`.`id`)) ORDER BY `al`.`created_at` DESC LIMIT 0, 100 ;

-- --------------------------------------------------------

--
-- Structure for view `user_dashboard_stats`
--
DROP TABLE IF EXISTS `user_dashboard_stats`;

CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW `user_dashboard_stats`  AS SELECT `u`.`id` AS `user_id`, `u`.`username` AS `username`, `u`.`role` AS `role`, count(`ph`.`id`) AS `total_pdfs_generated`, max(`ph`.`created_at`) AS `last_pdf_generated`, count(`al`.`id`) AS `total_activities` FROM ((`users` `u` left join `pdf_history` `ph` on(`u`.`id` = `ph`.`user_id`)) left join `activity_logs` `al` on(`u`.`id` = `al`.`user_id`)) WHERE `u`.`is_active` = 1 GROUP BY `u`.`id`, `u`.`username`, `u`.`role` ;

--
-- Indexes for dumped tables
--

--
-- Indexes for table `activity_logs`
--
ALTER TABLE `activity_logs`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_user_id` (`user_id`),
  ADD KEY `idx_action` (`action`),
  ADD KEY `idx_created_at` (`created_at`),
  ADD KEY `idx_ip_address` (`ip_address`),
  ADD KEY `idx_activity_user_action` (`user_id`,`action`);

--
-- Indexes for table `email_notifications`
--
ALTER TABLE `email_notifications`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_user_id` (`user_id`),
  ADD KEY `idx_recipient` (`recipient_email`),
  ADD KEY `idx_sent` (`sent`),
  ADD KEY `idx_created_at` (`created_at`);

--
-- Indexes for table `login_attempts`
--
ALTER TABLE `login_attempts`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_username` (`username`),
  ADD KEY `idx_ip_address` (`ip_address`),
  ADD KEY `idx_success` (`success`),
  ADD KEY `idx_attempt_time` (`attempt_time`),
  ADD KEY `idx_login_attempts_user_time` (`username`,`attempt_time`);

--
-- Indexes for table `password_reset_tokens`
--
ALTER TABLE `password_reset_tokens`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `token` (`token`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `pdf_history`
--
ALTER TABLE `pdf_history`
  ADD PRIMARY KEY (`id`),
  ADD KEY `idx_pdf_history_user_date` (`user_id`,`created_at`);

--
-- Indexes for table `rooms_config`
--
ALTER TABLE `rooms_config`
  ADD PRIMARY KEY (`id`),
  ADD KEY `created_by` (`created_by`),
  ADD KEY `idx_room_number` (`room_number`),
  ADD KEY `idx_building` (`building`),
  ADD KEY `idx_active_rooms` (`is_active`);

--
-- Indexes for table `settings`
--
ALTER TABLE `settings`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `setting_key` (`setting_key`),
  ADD KEY `updated_by` (`updated_by`);

--
-- Indexes for table `students`
--
ALTER TABLE `students`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `roll_number` (`roll_number`),
  ADD KEY `idx_students_branch_detained` (`branch`,`is_detained`),
  ADD KEY `idx_students_branch_section_detained` (`branch`,`section`,`is_detained`);

--
-- Indexes for table `system_settings`
--
ALTER TABLE `system_settings`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `setting_key` (`setting_key`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`),
  ADD UNIQUE KEY `email` (`email`),
  ADD KEY `idx_users_role_active` (`role`,`is_active`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `activity_logs`
--
ALTER TABLE `activity_logs`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=361;

--
-- AUTO_INCREMENT for table `email_notifications`
--
ALTER TABLE `email_notifications`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `login_attempts`
--
ALTER TABLE `login_attempts`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `password_reset_tokens`
--
ALTER TABLE `password_reset_tokens`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `pdf_history`
--
ALTER TABLE `pdf_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=11;

--
-- AUTO_INCREMENT for table `rooms_config`
--
ALTER TABLE `rooms_config`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `settings`
--
ALTER TABLE `settings`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `students`
--
ALTER TABLE `students`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=1939;

--
-- AUTO_INCREMENT for table `system_settings`
--
ALTER TABLE `system_settings`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `activity_logs`
--
ALTER TABLE `activity_logs`
  ADD CONSTRAINT `activity_logs_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL;

--
-- Constraints for table `email_notifications`
--
ALTER TABLE `email_notifications`
  ADD CONSTRAINT `email_notifications_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE SET NULL;

--
-- Constraints for table `password_reset_tokens`
--
ALTER TABLE `password_reset_tokens`
  ADD CONSTRAINT `password_reset_tokens_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `pdf_history`
--
ALTER TABLE `pdf_history`
  ADD CONSTRAINT `pdf_history_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `rooms_config`
--
ALTER TABLE `rooms_config`
  ADD CONSTRAINT `rooms_config_ibfk_1` FOREIGN KEY (`created_by`) REFERENCES `users` (`id`);

--
-- Constraints for table `settings`
--
ALTER TABLE `settings`
  ADD CONSTRAINT `settings_ibfk_1` FOREIGN KEY (`updated_by`) REFERENCES `users` (`id`) ON DELETE SET NULL;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
