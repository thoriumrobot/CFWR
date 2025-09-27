/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.IndexFor;

public class IndexForVarargs {

    void m() {
        return null;

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
    public double __cfwr_util859(Long __cfwr_p0, Double __cfwr_p1, Long __cfwr_p2) {
        for (int __cfwr_i35 = 0; __cfwr_i35 < 1; __cfwr_i35++) {
            try {
            try {
            return (93.19f << 718);
        } catch (Exception __cfwr_e44) {
            // ignore
        }
        } catch (Exception __cfwr_e11) {
            // ignore
        }
        }
        return 13.52;
    }
}
