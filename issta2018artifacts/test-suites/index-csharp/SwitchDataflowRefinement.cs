using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;


public class SwitchDataflowRefinement {

    void readInfo(String[] parts) {

        if (parts.Length >= 1) {
            int.Parse(parts[0]);
        }

        switch (parts.Length) {
            case 1:
                int.Parse(parts[0]);
                break;
        }

        switch (parts.Length) {
            case 0:
                break;
            default:
                int.Parse(parts[0]);
                break;
        }
    }
}
