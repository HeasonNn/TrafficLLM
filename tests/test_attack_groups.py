import unittest

from scripts.attack_groups import get_dataset_categories, match_attack_category


class AttackGroupsTaxonomyTest(unittest.TestCase):
    def test_cic_ids2017_exposes_only_requested_categories(self):
        self.assertEqual(
            get_dataset_categories("CIC-IDS2017"),
            ["Bot", "Brute Force", "DoS/DDoS", "Web Attack"],
        )
        self.assertEqual(match_attack_category("CIC-IDS2017", "Botnet_ARES"), "Bot")
        self.assertEqual(match_attack_category("CIC-IDS2017", "FTP_Patator"), "Brute Force")
        self.assertEqual(match_attack_category("CIC-IDS2017", "DoS_Hulk"), "DoS/DDoS")
        self.assertEqual(match_attack_category("CIC-IDS2017", "XSS"), "Web Attack")

    def test_cic_ids2017_non_target_families_are_not_emitted(self):
        self.assertIsNone(match_attack_category("CIC-IDS2017", "Port_Scan"))
        self.assertIsNone(match_attack_category("CIC-IDS2017", "Infiltration"))

    def test_hypervision_matches_requested_categories(self):
        self.assertEqual(
            get_dataset_categories("HyperVision"),
            ["Brute Force", "DoS/DDoS", "Malware", "Scan", "Web Attack"],
        )
        self.assertEqual(match_attack_category("HyperVision", "sshpwdla"), "Brute Force")
        self.assertEqual(match_attack_category("HyperVision", "charrdos"), "DoS/DDoS")
        self.assertEqual(match_attack_category("HyperVision", "emotet"), "Malware")
        self.assertEqual(match_attack_category("HyperVision", "sshscan"), "Scan")
        self.assertEqual(match_attack_category("HyperVision", "csrf"), "Web Attack")

    def test_unsw_nb15_matches_requested_categories(self):
        self.assertEqual(
            get_dataset_categories("UNSW-NB15"),
            ["DoS/DDoS", "Malware", "Generic", "Scan"],
        )
        self.assertEqual(match_attack_category("UNSW-NB15", "DoS"), "DoS/DDoS")
        self.assertEqual(match_attack_category("UNSW-NB15", "Shellcode"), "Malware")
        self.assertEqual(match_attack_category("UNSW-NB15", "Generic"), "Generic")
        self.assertEqual(match_attack_category("UNSW-NB15", "Reconnaissance"), "Scan")

    def test_cic_iiot2025_matches_requested_categories(self):
        self.assertEqual(
            get_dataset_categories("CIC-IIoT2025"),
            ["Brute Force", "DoS/DDoS", "MITM", "Malware", "Scan", "Web Attack"],
        )
        self.assertEqual(match_attack_category("CIC-IIoT2025", "bruteforce_dictionary-ssh"), "Brute Force")
        self.assertEqual(match_attack_category("CIC-IIoT2025", "ddos_udp-flood"), "DoS/DDoS")
        self.assertEqual(match_attack_category("CIC-IIoT2025", "mitm_arp-spoofing"), "MITM")
        self.assertEqual(match_attack_category("CIC-IIoT2025", "malware_mirai-udp-flood"), "Malware")
        self.assertEqual(match_attack_category("CIC-IIoT2025", "recon_port-scan"), "Scan")
        self.assertEqual(match_attack_category("CIC-IIoT2025", "web_xss"), "Web Attack")

    def test_dohbrw_is_split_per_tool(self):
        self.assertEqual(get_dataset_categories("DoHBrw"), ["DNS2TCP", "DNSCat2", "Iodine"])
        self.assertEqual(match_attack_category("DoHBrw", "dns2tcp"), "DNS2TCP")
        self.assertEqual(match_attack_category("DoHBrw", "dnscat2"), "DNSCat2")
        self.assertEqual(match_attack_category("DoHBrw", "iodine"), "Iodine")

    def test_cicapt_iiot2024_is_split_into_tactics(self):
        self.assertEqual(
            get_dataset_categories("CICAPT-IIoT2024"),
            ["Cleanup", "Collection", "CC", "CA", "Discovery", "Exfiltration", "Lateral Move", "Persistence"],
        )
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "cleanup"), "Cleanup")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "collection"), "Collection")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "command_and_control"), "CC")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "credential_access"), "CA")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "discovery"), "Discovery")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "exfiltration"), "Exfiltration")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "lateral_movement"), "Lateral Move")
        self.assertEqual(match_attack_category("CICAPT-IIoT2024", "persistence"), "Persistence")


if __name__ == "__main__":
    unittest.main()
