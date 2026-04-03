#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attack category grouping definitions for TrafficLLM UAD evaluation.

This module provides grouping dictionaries and matching logic for
mapping dataset attack files to their attack categories.
"""

import re
from typing import Optional, Dict, List

GROUPS_CICIDS2017: Dict[str, List[str]] = {
    "Bot": ["Botnet_ARES"],
    "Brute Force": ["Brute_Force", "FTP_Patator", "SSH_Patator"],
    "DoS/DDoS": ["DDoS_LOIT", "DoS_GoldenEye", "DoS_Hulk", "DoS_Slowhttptest", "DoS_slowloris"],
    "Web Attack": ["XSS"],
}

GROUPS_CICIIOT2025: Dict[str, List[str]] = {
    "Brute Force": ["bruteforce_dictionary-ssh", "bruteforce_dictionary-telnet"],
    "DoS/DDoS": [
        "dos_syn-flood", "dos_tcp-flood", "dos_udp-flood", "dos_http-flood",
        "dos_icmp-flood", "dos_slowloris", "dos_connect-flood", "dos_push-ack-flood",
        "dos_ack-frag-flood", "dos_icmp-frag-flood", "dos_mqtt-publish-flood",
        "dos_rst-fin-flood", "dos_synonymousip-flood", "dos_udp-frag-flood",
        "ddos_syn-flood", "ddos_tcp-flood", "ddos_udp-flood", "ddos_http-flood",
        "ddos_icmp-flood", "ddos_slowloris", "ddos_connect-flood", "ddos_push-ack-flood",
        "ddos_ack-frag-flood", "ddos_icmp-frag-flood", "ddos_mqtt-publish-flood",
        "ddos_rst-fin-flood", "ddos_synonymousip-flood", "ddos_udp-frag-flood"
    ],
    "MITM": ["mitm_arp-spoofing", "mitm_impersonation", "mitm_ip-spoofing"],
    "Malware": ["malware_mirai-syn-flood", "malware_mirai-udp-flood"],
    "Scan": [
        "recon_host-disc-arp-ping", "recon_host-disc-tcp-ack-ping", "recon_host-disc-tcp-syn-ping",
        "recon_host-disc-tcp-syn-stealth", "recon_host-disc-udp-ping", "recon_os-scan",
        "recon_ping-sweep", "recon_port-scan", "recon_vuln-scan"
    ],
    "Web Attack": ["web_backdoor-upload", "web_command-injection", "web_sql-injection-blind", "web_sql-injection", "web_xss"],
}

GROUPS_UNSW_NB15: Dict[str, List[str]] = {
    "DoS/DDoS": ["DoS"],
    "Malware": ["Exploits", "Backdoor", "Worms", "Shellcode"],
    "Generic": ["Generic"],
    "Scan": ["Reconnaissance", "Analysis", "Fuzzers"],
}

GROUPS_HYPERVISION: Dict[str, List[str]] = {
    "Brute Force": ["telnetpwdla", "telnetpwdmd", "telnetpwdsm", "sshpwdla", "sshpwdmd", "sshpwdsm"],
    "DoS/DDoS": ["charrdos", "cldaprdos", "dnsrdos", "memcachedrdos", "ntprdos", "riprdos", "ssdprdos", "synsdos", "udpsdos", "icmpsdos", "crossfirela", "crossfiremd", "crossfiresm", "rstsdos", "lrtcpdos02", "lrtcpdos05", "lrtcpdos10"],
    "Malware": ["agentinject", "adload", "bitcoinminer", "coinminer", "dridex", "emotet", "koler", "magic", "mazarbot", "mobidash", "paraminject", "plankton", "ransombo", "sality", "snojan", "svpeng", "thbot", "trickbot", "trojanminer", "wannalocker", "webcompanion", "zsone", "spam1", "spam50", "spam100", "ccleaner", "persistence", "oracle", "penetho", "trickster"],
    "Scan": ["ackport", "dns_lrscan", "http_lrscan", "icmp_lrscan", "ipidaddr", "ipidport", "netbios_lrscan", "rdp_lrscan", "sslscan", "httpscan", "httpsscan", "icmpscan", "dnsscan", "ntpscan", "scrapy", "sshscan", "telnet_lrscan", "vlc_lrscan", "sqlscan"],
    "Web Attack": ["xss", "codeinject", "webshell", "csrf", "csfr"],
}

GROUPS_DOHBRW: Dict[str, List[str]] = {
    "DNS2TCP": ["dns2tcp"],
    "DNSCat2": ["dnscat2"],
    "Iodine": ["iodine"],
}

GROUPS_CIC_APT_IIoT2024: Dict[str, List[str]] = {
    "Cleanup": ["cleanup"],
    "Collection": ["collection"],
    "CC": ["command_and_control", "command_control", "cc"],
    "CA": ["credential_access", "credentialaccess", "ca"],
    "Discovery": ["discovery"],
    "Exfiltration": ["exfiltration"],
    "Lateral Move": ["lateral_movement", "lateral_move"],
    "Persistence": ["persistence"],
}

DATASET_ALIASES: Dict[str, str] = {
    "cicids2017": "cicids2017",
    "ids2017": "cicids2017",
    "cic-ids2017": "cicids2017",
    "ciciiot2025": "ciciiot2025",
    "cic-iiot2025": "ciciiot2025",
    "unsw_nb15": "unsw_nb15",
    "unsw-nb15": "unsw_nb15",
    "unswnb15": "unsw_nb15",
    "hypervision": "hypervision",
    "dohbrw": "dohbrw",
    "dohbrw_smoke": "dohbrw",
    "dohbrw_subset10k_source": "dohbrw",
    "cic_apt_iiot2024": "cic_apt_iiot2024",
    "cicapt-iiot2024": "cic_apt_iiot2024",
    "cicapt_iiot2024": "cic_apt_iiot2024",
}

def normalize_dataset_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset.lower().strip(), dataset.lower().strip())

def _norm_token(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()

def _get_group_dict(dataset: str) -> Optional[Dict[str, List[str]]]:
    normalized = normalize_dataset_name(dataset)
    if normalized == "cicids2017":
        return GROUPS_CICIDS2017
    if normalized == "ciciiot2025":
        return GROUPS_CICIIOT2025
    if normalized == "unsw_nb15":
        return GROUPS_UNSW_NB15
    if normalized == "hypervision":
        return GROUPS_HYPERVISION
    if normalized == "dohbrw":
        return GROUPS_DOHBRW
    if normalized == "cic_apt_iiot2024":
        return GROUPS_CIC_APT_IIoT2024
    return None

def match_attack_category(dataset: str, file_token: str) -> Optional[str]:
    groups = _get_group_dict(dataset)
    if groups is None:
        return None
    tok = _norm_token(file_token)
    for category, keys in groups.items():
        for key in keys:
            key_norm = _norm_token(key)
            if len(key_norm) <= 2:
                if tok == key_norm:
                    return category
                continue
            if key_norm in tok or tok in key_norm:
                return category
    return None

def get_all_categories() -> List[str]:
    all_categories = set()
    for groups in [GROUPS_CICIDS2017, GROUPS_CICIIOT2025, GROUPS_UNSW_NB15, GROUPS_HYPERVISION, GROUPS_DOHBRW, GROUPS_CIC_APT_IIoT2024]:
        all_categories.update(groups.keys())
    return sorted(all_categories)

def get_dataset_categories(dataset: str) -> List[str]:
    groups = _get_group_dict(dataset)
    if groups is None:
        return []
    return list(groups.keys())

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python attack_groups.py <dataset> <file_token>")
        print("\nExample:")
        print("  python attack_groups.py CICIDS2017 DoS_Hulk")
        print("  python attack_groups.py hypervision dns_lrscan")
        print("\nAvailable datasets: CICIDS2017, CICIIOT2025, UNSW_NB15, hypervision, DoHBrw, CIC_APT_IIoT2024")
        print(f"\nAll categories: {get_all_categories()}")
        sys.exit(1)
    dataset = sys.argv[1]
    file_token = sys.argv[2]
    category = match_attack_category(dataset, file_token)
    print(f"Dataset: {dataset}")
    print(f"File: {file_token}")
    print(f"Category: {category}")
