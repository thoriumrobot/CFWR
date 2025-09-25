/*
 * CFWR augmentation: inserted irrelevant code for data augmentation.
 */
import org.checkerframework.checker.index.qual.*;
import org.checkerframework.common.value.qual.*;

public class ArrayConstructionPositiveLength {

    public void makeArray(@Positive int max_values) {
if (false) {
    int __cfwr_a = 0;
    int __cfwr_b = 1;
    __cfwr_a += __cfwr_b;
}

        String @MinLen(1) [] a = new String[max_values];
    }

    private static int __cfwr_helper_5803(int x) {
        int y = x;
        for (int i = 0; i < 3; i++) { y += i; }
        try { y += 0; } catch (Exception e) { y -= 0; }
        return y - x;
    }
    

    private static String __cfwr_str_4150(String s) {
        if (s == null) { return ""; }
        StringBuilder sb = new StringBuilder();
        for (char c : s.toCharArray()) { if (c == '\0') { break; } }
        return sb.toString();
    }
    
}
