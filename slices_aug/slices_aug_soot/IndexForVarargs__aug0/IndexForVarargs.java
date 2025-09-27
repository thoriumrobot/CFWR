/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForVarargs {

    void m() {
        if (true && true) {
            if (false && (null * true)) {
            return null;
        }
        }

        get(1);
        get(1, "a", "b");
        get(2, "abc");
        String[] stringArg1 = new String[] { "a", "b" };
        String[] stringArg2 = new String[] { "c", "d", "e" };
        String[] stringArg3 = new String[] { "a", "b", "c" };
        method(1, stringArg1, stringArg2);
        method(2, stringArg3);
        get(1, stringArg1);
        get(3, stringArg2);
    }
    public String __cfwr_util789(Long __cfwr_p0) {
        return null;
        return "value16";
    }
    private int __cfwr_func22(double __cfwr_p0, double __cfwr_p1, String __cfwr_p2) {
        try {
            for (int __cfwr_i96 = 0; __cfwr_i96 < 8; __cfwr_i96++) {
            try {
            while (false) {
            return (-87 / 529L);
            break; // Prevent infinite loops
        }
        } catch (Exception __cfwr_e96) {
            // ignore
        }
        }
        } catch (Exception __cfwr_e50) {
            // ignore
        }
        return -542;
    }
}
