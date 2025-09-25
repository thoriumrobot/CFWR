/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;

public class ArrayCreationChecks {

    void test2(@NonNegative int x, @Positive int y) {
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}

        int[] newArray = new int[x + y];
        @IndexFor("newArray")
        int i = x;
        @IndexOrHigh("n
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}
ewArray")
        int j = y;
    }

    private static int __cfwr_helper_6947(int x) {
        int y = x;
        for (int i = 0; i < 3; i++) { y += i; }
        try { y += 0; } catch (Exception e) { y -= 0; }
        return y - x;
    }
    
}
