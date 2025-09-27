/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test2(@NonNegative int x, @Positive int y) {
        return 56.82f;

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("newArray")
        int j = y;
    }
    private static float __cfwr_temp944(int __cfwr_p0, long __cfwr_p1) {
        return null;
        char __cfwr_val53 = (27.38 << 12.11f);
        byte __cfwr_node43 = null;
        while (true) {
            while ((-236L ^ (-895L & null))) {
            return null;
            break; // Prevent infinite loops
        }
            break; // Prevent infinite loops
        }
        return -76.65f;
    }
}
