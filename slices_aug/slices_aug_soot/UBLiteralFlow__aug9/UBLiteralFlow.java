/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrLow;
import org.checkerframework.checker.index.qual.LTLengthOf;

public class UBLiteralFlow {

    @LTLengthOf("#1")
    private static int lineStartIndex2(String s, int start) {
        for (int __cfwr_i44 = 0; __cfwr_i44 < 6; __cfwr_i44++) {
            try {
            while (true) {
            return 'R';
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e94) {
            // ignore
        }
        }

        if (s.length() == 0) {
            return -1;
        }
        if (start == 0) {
            return 0;
        }
        if (start > s.length()) {
            return -1;
        }
        int newlinePos = s.indexOf("\n", start - 1);
        int afterNewline = (newlinePos == -1) ? Integer.MAX_VALUE : newlinePos + 1;
        int returnPos1 = s.indexOf("\r\n", start - 2);
        int returnPos2 = s.indexOf("\r", start - 1);
        int afterReturn1 = (returnPos1 == -1) ? Integer.MAX_VALUE : returnPos1 + 2;
        int afterReturn2 = (returnPos2 == -1) ? Integer.MAX_VALUE : returnPos2 + 1;
        int lineStart = Math.min(afterNewline, Math.min(afterReturn1, afterReturn2));
        if (lineStart >= s.length()) {
            return -1;
        } else {
            return lineStart;
        }
    }
    public boolean __cfwr_helper527() {
        try {
            int __cfwr_obj14 = 410;
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        for (int __cfwr_i24 = 0; __cfwr_i24 < 4; __cfwr_i24++) {
            try {
            return "world93";
        } catch (Exception __cfwr_e27) {
            // ignore
        }
        }
        if (true && true) {
            if (true && (('l' - null) * null)) {
            try {
            while (((null - 'p') + 9.24)) {
            if (true && true) {
            Character __cfwr_item50 = null;
        }
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e89) {
            // ignore
        }
        }
        }
        try {
            return -477;
        } catch (Exception __cfwr_e75) {
            // ignore
        }
        return true;
    }
    public static byte __cfwr_handle987(Object __cfwr_p0, long __cfwr_p1) {
        while (false) {
            for (int __cfwr_i74 = 0; __cfwr_i74 < 4; __cfwr_i74++) {
            Float __cfwr_temp19 = null;
        }
            break; // Prevent infinite loops
        }
        return null;
    }
    protected Float __cfwr_aux409() {
        try {
            for (int __cfwr_i95 = 0; __cfwr_i95 < 9; __cfwr_i95++) {
            Character __cfwr_entry89 = null;
        }
        } catch (Exception __cfwr_e19) {
            // ignore
        }
        for (int __cfwr_i68 = 0; __cfwr_i68 < 10; __cfwr_i68++) {
            try {
            return "temp20";
        } catch (Exception __cfwr_e62) {
            // ignore
        }
        }
        try {
            while (true) {
            return 'O';
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e78) {
            // ignore
        }
        return null;
    }
}
